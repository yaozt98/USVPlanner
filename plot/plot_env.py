import os.path

import matplotlib.pyplot as plt
import numpy as np
import glob
import matplotlib as mpl
import math
import matplotlib.cm as cm
import pyclipper


colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
lines = ['-','-',':',':']
markers=['^','v','.','*','^']
laser_max_range = 100.0
def zoom_contour(contour,margin):
    pco = pyclipper.PyclipperOffset()
    pco.MiterLimit = 15
    pco.AddPath(contour,pyclipper.JT_MITER,pyclipper.ET_CLOSEDPOLYGON)
    solution = pco.Execute(margin)
    solution = np.array(solution).reshape(-1,2).astype(float)
    return solution

def draw_rotated_boat(center, rotation_angle):
    # the shape of usv and boat
    bias =np.array([[0, 13],
            [2, 12.6], [4, 11.25], [6, 8.6], [8, 0],
            [8, -4.6], [7.9, -8.9], [7.5, -15.4], [7, -21], [6, -23.5], [5, -25.9], [4, -26.7], [2, -27],
            [0, -27.5],
            [-2, -27], [-4, -26.7], [-5, -25.9], [-6, -23.5], [-7, -21], [-7.5, -15.4], [-7.9, -8.9], [-8, -4.6],
            [-8, 0], [-6, 8.6], [-4, 11.25], [-2, 12.6]])

    polygon_x = bias[:,0] + center[0]
    polygon_y = bias[:,1] + center[1]
    polygon_points = list(zip(polygon_x, polygon_y))

    rotation_polygon = []
    angle_radians = rotation_angle
    for point in polygon_points:
        x_rotate = (point[0] - center[0]) * np.cos(angle_radians) - (point[1] - center[1]) * np.sin(angle_radians) + center[0]
        y_rotate = (point[0] - center[0]) * np.sin(angle_radians) + (point[1] - center[1]) * np.cos(angle_radians) + center[1]
        rotation_polygon.append((x_rotate, y_rotate))
    return rotation_polygon

def draw_ego_boat(ax, usv, start=False):
    if start:
        t=0
        r_angle=0
    else:
        t = len(usv)-1
        r_angle = np.arctan2(usv[t, 1] - usv[t-1, 1],
                             usv[t, 0] - usv[t-1, 0]) - np.pi / 2
    rx_center, ry_center = usv[t]
    r_points = draw_rotated_boat(np.asarray([rx_center, ry_center]), r_angle)
    r_points = zoom_contour(r_points,2)
    r_ship = plt.Polygon(r_points, closed=True,edgecolor='black', facecolor='#00ffff')
    plt.text(rx_center + 10, ry_center - 5, str(t) + ' s', fontsize=15)
    ax.add_patch(r_ship)

def draw_obstacle_boat(ax, obstacles, boat_index, angle, zoom_in,start=False):
    if start:
        t=2
        k=1
        al = 0.3
    else:
        t=min(len(obstacles),150)-1
        end_point = obstacles[t, boat_index, :]
        k=0
        al=1
        for i in range(t):
            next_end_point = obstacles[t-i-1, boat_index, :]
            if next_end_point[0] != end_point[0] or next_end_point[1] != end_point[1]:
                k=i+1
                break
    x_center, y_center = obstacles[t, boat_index, :]
    rotation_angle = np.arctan2(obstacles[t, boat_index, 1] - obstacles[t - k, boat_index, 1],
                                obstacles[t, boat_index, 0] - obstacles[t - k, boat_index, 0]) - np.pi / 2
    boat_points = draw_rotated_boat(np.asarray([x_center, y_center]), rotation_angle)
    boat_points = zoom_contour(boat_points,zoom_in)
    boat = plt.Polygon(boat_points, closed=True, edgecolor='black', facecolor=colors[boat_index],alpha=al)
    ax.add_patch(boat)
    if start:
        t = t-2
        if -30 <= angle <= 30 or -180 <= angle <= -150 or 150 <= angle <= 180:
            plt.text(x_center - 5, y_center - 22, str(t) + ' s', fontsize=15)
        elif angle == -90:
            plt.text(x_center + 10, y_center+5, str(t) + ' s', fontsize=15)
        else:
            plt.text(x_center + 10, y_center - 15, str(t) + ' s', fontsize=15)

def rotate(x,y,k):
    theta = math.radians(k)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    x_new = x * cos_theta + y * sin_theta
    y_new = -x * sin_theta + y * cos_theta
    return x_new,y_new

def velocity_plot(path_dir,file_names):
    plt.figure(figsize=(12,5))
    for i,file in enumerate(file_names):
        log_env = np.load(os.path.join(path_dir,file))
        usv = log_env['usv']
        velocity = usv[:,2]
        name=file[:-4]
        x_axis = np.arange(len(velocity))
        plt.plot(x_axis,velocity,label=name,color=colors[i],ls=lines[i],marker=markers[i],zorder=10-i)
    plt.grid()
    plt.xlabel('Time steps', fontsize=14)
    plt.ylabel('Velocity (m/s)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.legend(loc='lower right')
    # plt.savefig("Velocity.svg", dpi=600, bbox_inches='tight')
    # plt.savefig("Velocity.png", dpi=300, bbox_inches='tight')
    plt.show()


def heading_plot(path_dir,file_names):
    plt.figure(figsize=(8, 5))
    for i,file in enumerate(file_names):
        log_env = np.load(os.path.join(path_dir,file))
        usv = log_env['usv']
        heading = usv[:,3] * 180 / np.pi
        heading[heading<0] = heading[heading<0] +360
        name=file[:-4]
        x_axis = np.arange(len(heading))
        plt.plot(x_axis,heading,label=name,color=colors[i],ls=lines[i],marker=markers[i])
    plt.grid()
    plt.xlabel('Time steps', fontsize=14)
    plt.ylabel('Heading (\u00B0)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.legend(loc='lower right')
    # plt.savefig("Heading.svg", dpi=600, bbox_inches='tight')
    # plt.savefig("Heading.png", dpi=300, bbox_inches='tight')
    plt.show()


def lidar_plot(path_directory):
    cmap = cm.Blues(np.linspace(0.3, 0.9, 20))
    cmap = mpl.colors.ListedColormap(cmap[10:, :-1])
    file_name = sorted(glob.glob(path_directory + '/*.npz'))
    laser_beam=1800
    for file in file_name:
        log_env = np.load(file)
        usv = log_env['usv']
        steps = usv.shape[0]
        laser = log_env['laser']

        f_ego = 30
        for t in range(f_ego, steps - f_ego, f_ego):
            fig, ax = plt.subplots(figsize=(7,7))
            plt.tick_params(labelsize=20)
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname("serif") for label in labels]
            ax.set_xlim([(-laser_max_range - 1) / 20, (laser_max_range + 1) / 20])
            ax.set_ylim([(-laser_max_range - 1) / 20, (laser_max_range + 1) / 20])
            ax.set_aspect('equal')
            ax.set_title('LiDAR Reflections', fontsize=20)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            lines = []
            color_map=np.linspace(0,0.8,60)
            cmap = plt.get_cmap('magma')
            color_map = cmap(color_map)
            for laser_i in range(laser_beam):
                px = laser[t][laser_i][2] - laser[t][laser_i][0]
                py = laser[t][laser_i][3] - laser[t][laser_i][1]
                if np.linalg.norm([px, py]) < laser_max_range:
                    dotx = laser[t][laser_i][2] - laser[t][laser_i][0]
                    doty = laser[t][laser_i][3] - laser[t][laser_i][1]
                    ax.plot(dotx / 30, doty / 30, color='red', marker='o', markersize=1)

                if np.linalg.norm([px,py]) >= laser_max_range:
                    A = np.array((laser[t][laser_i][0], laser[t][laser_i][1]))
                    B = np.array((laser[t][laser_i][2], laser[t][laser_i][3]))
                    AB = B - A
                    AB_unit = AB / np.linalg.norm(AB)
                    p_r = laser_max_range * AB_unit
                    px = p_r[0]
                    py = p_r[1]

                if laser_i % 30 == 0:
                    lines.append([(0, 0), (px / 30, py / 30)])

            lc = mpl.collections.LineCollection(lines, linewidths=1.5,colors=color_map, alpha=0.7)
            ax.add_collection(lc)
            # save_name = path_directory + '/' + str(t) + '.svg'
            # plt.savefig(save_name, format='svg')
            plt.show()
    return

def show_scene(path_directory,background):
    file_name = sorted(glob.glob(path_directory + '/*.npz'))
    for file in file_name:
        log_env = np.load(file)
        usv = log_env['usv']
        steps = usv.shape[0]
        ships = log_env['ships']
        laser = log_env['laser']
        ship_num = ships.shape[1]
        goal = log_env['goal']
        radius = 15
        fig, ax = plt.subplots(figsize=(12, 12))
        # ax.yaxis.tick_right()
        # ax.xaxis.tick_top()
        # ax.set_ylabel('y/m', fontsize=20)
        # ax.set_xlabel('x/m', fontsize=20)
        ax.imshow(background,extent=[-500,500,-500,500])
        plt.tick_params(labelsize=20)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname("serif") for label in labels]
        ax.set_xlim(-500.0, 500.0)
        ax.set_ylim(-500.0, 500.0)
        ax.add_artist(plt.Circle(goal[0], 20, fill=True, color='#FF4040'))
        usv_traj=usv[:, 0:2]
        plt.plot(usv_traj[:,0], usv_traj[:,1], color='#00ffff',linewidth=2,zorder=1)
        draw_ego_boat(ax, usv_traj, start=True)
        draw_ego_boat(ax, usv_traj, start=False)
        obstacles = ships[:,:,0:2]
        obstacle_radius = ships[0, :, 2]
        for obs_index in range(ship_num):
            plt.plot(obstacles[:, obs_index,0], obstacles[:,obs_index,1], color=colors[obs_index])

        f_ego = 30
        for t in range(f_ego, steps - f_ego, f_ego):
            rx_center, ry_center = usv_traj[t]
            r_angle = np.arctan2(usv_traj[t, 1] - usv_traj[t - 1, 1],
                                 usv_traj[t, 0] - usv_traj[t - 1, 0]) - np.pi / 2
            r_points = draw_rotated_boat(np.asarray([rx_center, ry_center]), r_angle)
            r_points = zoom_contour(r_points, 2)
            r_ship = plt.Polygon(r_points, closed=True, edgecolor='black', facecolor='#00ffff')
            plt.text(rx_center+10, ry_center-15, str(t) + ' s', fontsize=15)
            ax.add_patch(r_ship)

        direction = []
        for boat_index in range(ship_num):
            x_start, y_start = obstacles[0, boat_index, :]
            x_end, y_end = obstacles[-1, boat_index, :]
            angle = int(math.atan2(y_end - y_start, x_end - x_start) * 180 / math.pi)
            direction.append(angle)

        for boat_index in range(ship_num):
            angle = direction[boat_index]
            zoom_in = (obstacle_radius[boat_index]-15)/2.5
            draw_obstacle_boat(ax, obstacles, boat_index, angle, zoom_in,start=True)
            draw_obstacle_boat(ax, obstacles, boat_index, angle, zoom_in)

        f_obs = f_ego
        last_step = min(steps-f_obs,150)
        for t in range(f_obs, last_step, f_obs):
            al = 0.3+0.5*t/last_step
            for boat_index in range(ship_num):
                x_center, y_center = obstacles[t, boat_index, :]
                rotation_angle = np.arctan2(obstacles[t, boat_index, 1]-obstacles[t-1, boat_index, 1], obstacles[t, boat_index, 0]-obstacles[t-1, boat_index, 0]) - np.pi/2
                boat_points = draw_rotated_boat(np.asarray([x_center, y_center]), rotation_angle)
                angle = direction[boat_index]
                zoom_in = (obstacle_radius[boat_index] - 15) / 2.5
                boat_points = zoom_contour(boat_points,zoom_in)
                boat = plt.Polygon(boat_points, closed=True, edgecolor='black', facecolor=colors[boat_index],alpha=al)
                ax.add_patch(boat)

        # ax.set_title("Collision avoidance based on proposed method", fontsize=14)
        plt.grid(True, which='both', color='#999999', linewidth=0.5)
        plt.minorticks_on()
        plt.grid(True, which='minor', color='#999999', linestyle='-', alpha=0.5)
        # plt.savefig(file[:-4] + '.svg', dpi=600, bbox_inches='tight')
        # plt.savefig(file[:-4] + '.png', dpi=300, bbox_inches='tight')
        # print('Saved!')
        plt.show()


if __name__ == "__main__":
    save_dir='/home/yao/project/USV_planner/plot/'
    background = plt.imread("background.png")

    # plot local path planning trajs
    show_scene(save_dir,background=background)

    # plot picture for lidar scan
    # lidar_plot(save_dir)

    # function: plot USV velocity and course changes under different planner
    # data = ["TD3.npz", "DDPG.npz","DWA.npz", "APF.npz"]
    # velocity_plot(save_dir,data)
    # heading_plot(save_dir, data)

