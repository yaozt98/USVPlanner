"""
the basic environment for DRL algorithm training
"""
import copy
import logging
import queue
import threading

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as mc
from numpy.linalg import norm
from utils.ship import Ship
from utils.usv import Usv
from utils.state import *
from policy.policy_factory import policy_factory
from info import *
from math import atan2, hypot, sqrt, cos, sin, fabs, inf, ceil
import time
from C_library.motion_plan_lib import *


class CrowdSim:
    def __init__(self, args, schedule:dict=None, e_mode=False):
        self.n_laser = args.lidar_dim
        self.laser_angle_resolute = args.laser_angle_resolute
        self.laser_min_range = args.laser_min_range
        self.laser_max_range = args.laser_max_range
        self.square_width = args.square_width
        self.half_square = args.square_width / 2
        self.discomfort_dist = args.discomfort_distance
        self.ship_policy_name = 'orca' # ship policy is fixed orca policy

        # last-time distance from the usv to the goal
        self.goal_distance_last = None
        self.usv_random=True
        self.e_mode = e_mode
        self.classical = args.classical

        # scan_intersection, each line connects the usv and the end of each laser beam
        self.scan_intersection = None # used for visualization

        # laser state
        self.scan_current = np.zeros(self.n_laser, dtype=np.float32)

        self.global_time = None
        self.time_limit = 300
        self.time_step = 1
        self.success_reward = 20
        self.collision_penalty = -10
        self.outside_penalty = 0
        self.timeout_penalty = -5
        self.discomfort_penalty_factor = 0.05
        self.goal_distance_factor = 0.08  # 0.01
        self.que = queue.Queue(2)
        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': 100, 'test': 500}

        margin = 600.0
        # the edge of local navigation area
        self.lines = [[(-margin, -margin), (-margin,  margin)], \
                        [(-margin,  margin), ( margin,  margin)], \
                        [( margin,  margin), ( margin, -margin)], \
                        [( margin, -margin), (-margin, -margin)]]
        self.circles = None # obstacles margin
        self.circle_radius = 400.0 # obstacle distribution margin
        self.ship_num = 0

        self.ships = None
        self.usv = Usv()
        self.usv.time_step = self.time_step
        self.angle_action=0

        self.case_counter = {'train': 0, 'test': 0, 'val': 0}
        print('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

        self.log_env = {}
        self.total_timesteps = 0
        self.schedule = schedule
        self.his=0
        self.ux_his = 0
        self.uy_his = -self.circle_radius

    def generate_random_ship_position(self):
        # initial min separation distance to avoid danger penalty at beginning
        self.ships = []
        for i in range(self.ship_num):
            self.ships.append(self.generate_circle_crossing_ship())

        for i in range(len(self.ships)):
            ship_policy = policy_factory[self.ship_policy_name]()
            ship_policy.time_step = self.time_step
            ship_policy.max_speed = self.ships[i].v_pref
            ship_policy.radius = self.ships[i].radius
            self.ships[i].set_policy(ship_policy)

    def generate_circle_crossing_ship(self):
        ship = Ship()
        ship.time_step = self.time_step

        ship.radius = np.int(np.random.uniform(15, 20, 1)[0])
        ship.v_pref = np.int(np.random.uniform(20, 40, 1)[0]) / 10 # velocity

        while True:
            # np.random.random generate a number within [0,1)
            px = np.random.uniform(-400, 400, 1)[0]
            py = np.random.uniform(-400, 400, 1)[0]
            while True:
                theta = np.random.random() * np.pi * 2
                if px > 0 and py > 0:
                    flag = theta >= 0 and theta < 0.5 * np.pi
                elif px <= 0 and py > 0:
                    flag = theta >= 0.5 * np.pi and theta < np.pi
                elif px <= 0 and py <= 0:
                    flag = theta >= np.pi and theta < 1.5 * np.pi
                else:
                    flag = theta >= 1.5 * np.pi and theta < 2 * np.pi
                if not flag:
                    break
            px_end = px + ship.v_pref * np.cos(theta) * 150
            py_end = py + ship.v_pref * np.sin(theta) * 150
            outside = px_end < -self.half_square or px_end > self.half_square or py_end > self.half_square or py_end < -self.half_square
            collide = False
            # avoid the newly generated obstacle too close to the start/end point of usv & existed obstacles
            for agent in [self.usv] + self.ships:
                min_dist = ship.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist or outside:
                    collide = True
                    break
            #         this break will retrigger the while True and go ship random generate
            if not collide:
                break
            #         this break will end while True and go ship.set
            # px, py, gx, gy, vx, vy, theta
        ship.set(px, py, px_end, py_end, 0, 0, theta)

        return ship

    def outside_check(self):
        outside = self.usv.px + self.usv.radius > self.half_square or\
                  self.usv.px - self.usv.radius < -self.half_square or\
                  self.usv.py + self.usv.radius > self.half_square or\
                  self.usv.py - self.usv.radius < -self.half_square
        return outside

    def get_lidar(self):
        scan = np.zeros(self.n_laser, dtype=np.float32)
        scan_end = np.zeros((self.n_laser, 2), dtype=np.float32)
        self.circles = np.zeros((self.ship_num, 3), dtype=np.float32)
        # here, more circles can be added to simulate obstacles
        for i in range(self.ship_num):
            self.circles[i, :] = np.array([self.ships[i].px, self.ships[i].py, self.ships[i].radius])
        usv_pose = np.array([self.usv.px, self.usv.py, self.usv.theta])
        num_line = len(self.lines)
        num_circle = self.ship_num
        InitializeEnv(num_line, num_circle, self.n_laser, self.laser_angle_resolute)
        for i in range (num_line):
            set_lines(4 * i    , self.lines[i][0][0])
            set_lines(4 * i + 1, self.lines[i][0][1])
            set_lines(4 * i + 2, self.lines[i][1][0])
            set_lines(4 * i + 3, self.lines[i][1][1])
        for i in range (num_circle):
            set_circles(3 * i    , self.ships[i].px)
            set_circles(3 * i + 1, self.ships[i].py)
            set_circles(3 * i + 2, self.ships[i].radius)
        set_robot_pose(usv_pose[0], usv_pose[1], usv_pose[2])
        cal_laser()
        self.scan_intersection = []
        for i in range(self.n_laser):
            scan[i] = get_scan(i) # length from usv to obstacle
            scan_end[i, :] = np.array([get_scan_line(4 * i + 2), get_scan_line(4 * i + 3)])
            ### used for visualization
            self.scan_intersection.append([(get_scan_line(4 * i + 0), get_scan_line(4 * i + 1)), \
                                           (get_scan_line(4 * i + 2), get_scan_line(4 * i + 3))])
            ### used for visualization

        self.scan_current = np.clip(scan, self.laser_min_range, self.laser_max_range) / self.laser_max_range
        ReleaseEnv()

    def reset(self, phase='test'):
        assert phase in ['train', 'val', 'test']
        self.global_time = 0
        self.log_env = {}
        # change obstacle number during training
        if not self.e_mode:
            if self.schedule is not None:
                steps = self.schedule["timesteps"]
                diffs = np.array(steps) - self.total_timesteps
                # find the interval the current timestep falls into
                idx = len(diffs[diffs <= 0]) - 1
                self.ship_num = self.schedule["num_obstacles"][idx]

        # random initialize the start point and goal of agent during training
        if not self.e_mode:
            if (self.total_timesteps <= self.schedule["timesteps"][-1]*1.5) and (int(self.total_timesteps / 2000) != self.his or self.total_timesteps==0):
                ux = np.int(np.random.uniform(-self.circle_radius, self.circle_radius, 1)[0])
                uy = np.int(np.random.uniform(-self.circle_radius, - self.circle_radius+20, 1)[0])
                self.his = int(self.total_timesteps / 1000)
                self.ux_his = ux
                self.uy_his = uy

        self.usv.set(self.ux_his,self.uy_his,-self.ux_his,-self.uy_his,5,0,np.pi / 2)
        # px, py, gx, gy, v, w, theta
        self.goal_distance_last = self.usv.get_goal_distance()
        self.generate_random_ship_position()
        self.get_lidar()

        # get the observation
        dx = self.usv.gx - self.usv.px
        dy = self.usv.gy - self.usv.py
        theta = self.usv.theta
        y_rel = dy * cos(theta) - dx * sin(theta)
        x_rel = dy * sin(theta) + dx * cos(theta)
        r = hypot(x_rel, y_rel) / self.square_width
        t = atan2(y_rel, x_rel) / np.pi
        ob_position = np.array([r, t], dtype=np.float32)
        self_state = FullState(self.usv.px, self.usv.py, self.usv.v, self.usv.w, self.usv.radius, \
                               self.usv.gx, self.usv.gy, self.usv.v_pref, self.usv.theta)
        ob_state = [pership.get_observable_state() for pership in self.ships]
        ob_coordinate = JointState(self_state, ob_state)
        self.log_env['usv'] = [np.array([self.usv.px, self.usv.py, self.usv.v, self.usv.theta])]
        self.log_env['goal'] = [np.array([self.usv.gx, self.usv.gy])]
        ships_position = []
        for pership in self.ships:
            ships_position.append(np.array([pership.px, pership.py, pership.radius, pership.vx]))
        self.log_env['ships'] = [np.array(ships_position)]
        self.log_env['reward'] = [np.zeros(1)]
        self.log_env['subreward'] = [np.array([0.0, 0.0, 0.0, 0.0])]
        lasers = []
        for laser in self.scan_intersection:
            lasers.append(np.array([laser[0][0], laser[0][1], laser[1][0], laser[1][1]]))
        self.log_env['laser'] = [np.array(lasers)]
        # return cost  # if calculate beam map efficiency
        if self.classical:
            return np.array(lasers),np.array([dx,dy]),np.array([self.usv.v, self.usv.theta])
        else:
            return self.scan_current, ob_position

    def step(self, action):
        ship_actions = []

        for ship in self.ships:
            if ship.reached_destination():
                ship_actions.append([0.0,0.0])
            else:
                ob = [other_ship.get_observable_state() for other_ship in self.ships if other_ship != ship]
                ship_actions.append(ship.act(ob))

        for i,ship_action in enumerate(ship_actions):
            self.ships[i].update_states(ship_action)

        # update states
        usv_x, usv_y, usv_theta = self.usv.compute_pose(action)
        self.usv.update_states(usv_x, usv_y, usv_theta, action)

        # get new laser scan and grid map
        self.get_lidar()
        self.global_time += self.time_step

        # if reaching goal
        goal_dist = hypot(usv_x - self.usv.gx, usv_y - self.usv.gy)
        reaching_goal = goal_dist < self.usv.radius

        # collision detection between the usv and ships
        collision = False
        dmin = (self.scan_current * self.laser_max_range).min()
        if dmin <= self.usv.radius:
            collision = True

        outside = self.outside_check()

        WR, GR, AR, CR = 0, 0, 0, 0
        if self.global_time >= self.time_limit - 1:
            reward = self.timeout_penalty
            done = True
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif outside:
            reward = self.outside_penalty
            done = True
            info = Outside()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        else:
            if ((dmin - self.usv.radius) < self.discomfort_dist):
                # penalize agent for getting too close
                WR = (dmin - self.usv.radius - self.discomfort_dist) * self.discomfort_penalty_factor
                done = False
                info = Danger(dmin)
            else:
                WR = 0
                done = False
                info = Nothing()

            # reward for goal reaching
            GR = self.goal_distance_factor * (self.goal_distance_last - goal_dist)

            # reward for angle keeping
            if self.angle_action * action[1] < 0:
                AR = -0.01
            else:
                AR = 0.01
            self.angle_action = action[1]

            # reward for following COLREG
            min_index = np.argmin(self.scan_current)
            if min_index > self.n_laser / 3 and min_index <= 37 * self.n_laser / 72:
                # reward for turn to port (left) side
                self.que.put([1, dmin])
                if self.que.full():
                    last_dist = self.que.get()
                    if last_dist[0] == 1 and last_dist[1] >= dmin:
                        CR = (np.pi / 2 - self.usv.theta) * np.exp(
                            - self.usv.v * cos(self.usv.theta) / self.usv.v_max) * 0.1

            elif min_index > 37 * self.n_laser / 72 and min_index < 13 * self.n_laser / 16:
                # reward for move forward and turn to starboard (right) side
                self.que.put([2, dmin])
                if self.que.full():
                    last_dist = self.que.get()
                    if last_dist[0] == 2 and last_dist[1] >= dmin:
                        if self.usv.theta <= np.pi / 2:
                            CR = self.usv.theta * np.exp(
                                self.usv.v * cos(self.usv.theta) / self.usv.v_max) * 0.05

            reward = WR + GR + AR + CR
            self.goal_distance_last = goal_dist

        # get the observation
        dx = self.usv.gx - self.usv.px
        dy = self.usv.gy - self.usv.py
        theta = self.usv.theta
        y_rel = dy * cos(theta) - dx * sin(theta)
        x_rel = dy * sin(theta) + dx * cos(theta)
        r = hypot(x_rel, y_rel) / self.square_width
        t = atan2(y_rel, x_rel) / np.pi
        ob_position = np.array([-r, -t], dtype=np.float32)
        self_state = FullState(self.usv.px, self.usv.py, self.usv.v, self.usv.w, self.usv.radius, \
                               self.usv.gx, self.usv.gy, self.usv.v_pref, self.usv.theta)
        ob_state = [pership.get_observable_state() for pership in self.ships]
        ob_coordinate = JointState(self_state, ob_state)
        self.log_env['usv'].append(np.array([self.usv.px, self.usv.py, self.usv.v, self.usv.theta]))
        self.log_env['goal'].append(np.array([self.usv.gx, self.usv.gy]))
        ships_position = []
        for pership in self.ships:
            ships_position.append(np.array([pership.px, pership.py, pership.radius, pership.vx]))
        self.log_env['ships'].append(np.array(ships_position))
        self.log_env['reward'].append(np.array([reward]))
        self.log_env['subreward'].append(np.array([WR, GR, AR, CR]))
        lasers = []
        for laser in self.scan_intersection:
            lasers.append(np.array([laser[0][0], laser[0][1], laser[1][0], laser[1][1]]))
        self.log_env['laser'].append(np.array(lasers))
        self.total_timesteps += 1
        if self.classical:
            return np.array(lasers),np.array([dx,dy]),np.array([self.usv.v, self.usv.theta]), reward, done, info
        else:
            return self.scan_current, ob_position, reward, done, info


    def render(self, mode='laser'):
        if mode == 'laser':
            self.ax.set_xlim(-500.0, 500.0)
            self.ax.set_ylim(-500.0, 500.0)
            for pership in self.ships:
                ship_circle = plt.Circle(pership.get_position(), pership.radius, fill=False, color='b')
                self.ax.add_artist(ship_circle)
            self.ax.add_artist(plt.Circle(self.usv.get_position(), self.usv.radius, fill=True, color='r'))
            plt.text(-450, -450, str(round(self.global_time, 2)), fontsize=20)
            x, y, theta = self.usv.px, self.usv.py, self.usv.theta
            dx = cos(theta)
            dy = sin(theta)
            self.ax.arrow(x, y, dx, dy,
                width=3,
                length_includes_head=True, 
                head_width=8,
                head_length=10,
                fc='r',
                ec='r')
            ii = 0
            lines = []
            while ii < self.n_laser:
                lines.append(self.scan_intersection[ii])
                ii = ii + 36
            lc = mc.LineCollection(lines)
            self.ax.add_collection(lc)
            plt.draw()
            plt.pause(0.001)
            plt.cla()

    def reset_with_eval_config(self,eval_config):
        self.log_env = {}
        self.global_time = 0
        self.scan_current = np.zeros(self.n_laser,dtype=np.float32)
        self.que = queue.Queue(2)
        # initialize the usv
        self.usv.set(0, -self.circle_radius, 0, self.circle_radius, 5, 0, np.pi / 2)
        # px, py, gx, gy, v, w, theta
        self.goal_distance_last = self.usv.get_goal_distance()
        # initialize the ships
        self.ships = []
        self.ship_num = len(eval_config["ships"]["px"])

        for i in range(self.ship_num):
            ship = Ship()
            px = eval_config["ships"]["px"][i]
            py = eval_config["ships"]["py"][i]
            px_end = eval_config["ships"]["px_end"][i]
            py_end = eval_config["ships"]["py_end"][i]
            ship.set(px, py, px_end, py_end, 0, 0, 0)
            ship.radius = eval_config["ships"]["radius"][i]
            ship.v_pref = eval_config["ships"]["v_pref"][i]
            ship.time_step = self.time_step
            ship_policy = policy_factory[self.ship_policy_name]()
            ship_policy.time_step = self.time_step
            ship_policy.max_speed = ship.v_pref
            ship_policy.radius = ship.radius
            ship.set_policy(ship_policy)
            self.ships.append(ship)

        self.get_lidar()

        # get the observation
        dx = self.usv.gx - self.usv.px
        dy = self.usv.gy - self.usv.py
        theta = self.usv.theta
        y_rel = dy * cos(theta) - dx * sin(theta)
        x_rel = dy * sin(theta) + dx * cos(theta)
        r = hypot(x_rel, y_rel) / self.square_width
        t = atan2(y_rel, x_rel) / np.pi
        ob_position = np.array([r, t], dtype=np.float32)
        self_state = FullState(self.usv.px, self.usv.py, self.usv.v, self.usv.w, self.usv.radius, \
                               self.usv.gx, self.usv.gy, self.usv.v_pref, self.usv.theta)
        ob_state = [pership.get_observable_state() for pership in self.ships]
        ob_coordinate = JointState(self_state, ob_state)
        self.log_env['usv'] = [np.array([self.usv.px, self.usv.py, self.usv.v, self.usv.theta])]
        self.log_env['goal'] = [np.array([self.usv.gx, self.usv.gy])]
        ships_position = []
        for pership in self.ships:
            ships_position.append(np.array([pership.px, pership.py, pership.radius, pership.vx]))
        self.log_env['ships'] = [np.array(ships_position)]
        self.log_env['reward'] = [np.zeros(1)]
        self.log_env['subreward'] = [np.array([0.0, 0.0, 0.0, 0.0])]
        lasers = []
        for laser in self.scan_intersection:
            lasers.append(np.array([laser[0][0], laser[0][1], laser[1][0], laser[1][1]]))
        self.log_env['laser'] = [np.array(lasers)]
        if self.classical:
            return np.array(lasers), np.array([dx, dy]), np.array([self.usv.v, self.usv.theta])
        else:
            return self.scan_current, ob_position
    def episode_data(self):
        episode = {}
        episode["ships"] = {}
        episode["ships"]["px"] = []
        episode["ships"]["py"] = []
        episode["ships"]["px_end"] = []
        episode["ships"]["py_end"] = []
        episode["ships"]["v_pref"] = []
        episode["ships"]["radius"] = []

        for ship in self.ships:
            episode["ships"]["px"].append(ship.px)
            episode["ships"]["py"].append(ship.py)
            episode["ships"]["px_end"].append(ship.gx)
            episode["ships"]["py_end"].append(ship.gy)
            episode["ships"]["v_pref"].append(ship.v_pref)
            episode["ships"]["radius"].append(ship.radius)

        return episode

    def clone(self):
        same_env = copy.deepcopy(self)
        same_env.lock = None
        return same_env
