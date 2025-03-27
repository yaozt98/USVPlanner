from utils.state import *
import numpy as np
from numpy.linalg import norm

class Ship():
    def __init__(self, mass=60, time_step=1):
        self.v_pref = 3.0
        self.radius = 15.0
        self.mass = mass
        self.iterations = 40
        self.policy = None
        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        # if using social force model, the control output is acceleration
        self.ax = 0.0
        self.ay = 0.0
        self.theta = None
        self.time_step = time_step

    def act(self, ob):
        """
        The state for ship is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    def set_policy(self, policy):
        self.policy = policy

    def sample_random_attributes(self):
        """
        Sample agent radius and v_pref attribute from certain distribution
        :return:
        """
        self.v_pref = np.random.uniform(0.5, 1.5)
        self.radius = np.random.uniform(0.3, 0.5)

    def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta
        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)
        
    def get_full_state(self):
        return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    def get_position(self):
        return self.px, self.py

    def get_goal_position(self):
        return self.gx, self.gy

    def get_goal_distance(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position()))

    def compute_position(self, action):
        px = self.px + action[0] * self.time_step
        py = self.py + action[1] * self.time_step
        return px, py

    def update_states(self, action, sfm=False):
        """
        Perform an action and update the state
        """
        if sfm:
            l_x = self.px
            l_y = self.py
            a_x = action[0]
            a_y = action[1]
            v0_x = self.vx
            v0_y = self.vy
            dt = self.time_step / self.iterations
            for i in range(self.iterations):
                l_x = (v0_x * dt + 0.5 * a_x * dt * dt) + l_x
                l_y = (v0_y * dt + 0.5 * a_y * dt * dt) + l_y
                v0_x = v0_x + a_x * dt 
                v0_y = v0_y + a_y * dt
            self.ax = a_x
            self.ay = a_y
            self.px = l_x
            self.py = l_y
            self.vx = v0_x
            self.vy = v0_y
               
        else:
            pos = self.compute_position(action)
            self.px, self.py = pos
            self.vx = action[0]
            self.vy = action[1]

    def reached_destination(self):
        return self.get_goal_distance() < self.radius
