from math import hypot,sin,cos,radians
import numpy as np
class Usv():
    def __init__(self):
        self.radius = 15
        self.v_pref = 3#5
        self.v_max = 6#10
        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.v = None
        self.w = None
        self.theta = None
        self.time_step = None

    def set(self, px, py, gx, gy, v, w, theta, v_pref=5):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.v = v
        self.w = w
        self.theta = theta
        if v_pref is not None:
            self.v_pref = v_pref

    def get_position(self):
        return self.px, self.py

    def compute_pose(self, action):
        theta_new = self.theta + np.pi/9 * action[1] * self.time_step
        speed_new = (self.v + action[0] * self.time_step).clip(0, self.v_max)
        px_new = self.px + speed_new * self.time_step * cos(theta_new)
        py_new = self.py + speed_new * self.time_step * sin(theta_new)
        return px_new, py_new, theta_new

    def get_goal_distance(self):
        return hypot(self.gx - self.px, self.gy - self.py)

    def update_states(self, px, py, theta, action):
        """
        Perform an action and update the state
        """
        self.px, self.py, self.theta = px, py, theta
        self.v = (self.v + action[0] * self.time_step).clip(0, self.v_max)
        self.w = action[1]
