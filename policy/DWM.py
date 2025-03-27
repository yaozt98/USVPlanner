import time
import math
import numpy as np
import matplotlib.pyplot as plt


# 定义状态类，用于存储船体航行过程中每一个时刻的状态----------
# time：状态对应的时刻，方便与ROS耦合
# x、y：船体的坐标
# velocity：速度
# yaw：偏航角，仿真只在二维空间，所以只定义一个角
# yawrate：角速度
# dyawrate：角加速度，代码中没有用到，在这里定义只是方便更加靠近真实的船
# cost：每走一步对应的成本
# --------------------------------------------------------
class state(object):
    def __init__(self, time, x, y, velocity, yaw, yawrate, dyawrate, cost):
        self.time = time
        self.x = x
        self.y = y
        self.velocity = velocity
        self.yaw = yaw
        self.yawrate = yawrate
        self.dyawrate = dyawrate
        self.cost = cost


# 定义航行过程中存储船所有状态和利用船的状态进行DWM的类-------
# ship：列表，列表的元素是上面定义的state类，用于存储航行过程中的所有时刻的状态
# obstacle：列表，元素是障碍点的坐标，这里的障碍点可以是建图完成之后再传入，或者通过传感器探测实时更新
# obstacle_windows：本意是定义一个船探测障碍物的探测半径，避障只需要考虑船能探测到的范围内进行就可以，本例中没有用到
# dt~saferadius：船的固有参数，比如maxvelocity是船能达到的最大速度。
class ship(object):
    def __init__(self):
        self.ship = []
        self.obstacle = np.array([[0, 0]])
        self.obstacle_windows = np.array([[0, 0]])
        self.dt = 0.1
        self.maxvelocity = 1.4
        self.minvelocity = 0
        self.maxlinearacc = 0.2
        self.maxdyawrate = 40 * math.pi / 180
        self.velocityres = 0.01
        self.yawrateres = 0.5 * math.pi / 180
        self.predicttime = 3
        self.to_goal_coeff = 1.0
        self.velocity_coeff = 1.0
        self.saferadius = 0.5
        self.goal = np.array([0, 0])
        self.arrive = False

    # 初始化函数----------------------------
    def initialState(self, state):
        self.ship.append(state)

    def initialobstacle(self, obstacle):
        self.obstacle = obstacle.assemble

    def initialgoal(self, goal):
        self.goal = goal

    # ------------------------------------
    # 运动函数，根据上面说的运动公式---------
    def motion(self, velocity, yawrate):
        temp_state = state(self.ship[-1].time + self.dt,
                           self.ship[-1].x + self.ship[-1].velocity * math.cos(self.ship[-1].yaw) * self.dt,
                           self.ship[-1].y + self.ship[-1].velocity * math.sin(self.ship[-1].yaw) * self.dt,
                           velocity,
                           self.ship[-1].yaw + yawrate * self.dt,
                           yawrate,
                           (yawrate - self.ship[-1].yawrate) / self.dt,
                           0)
        self.ship.append(temp_state)
        return temp_state

    # ------------------------------------
    # 动态窗口定义-------------------------
    def motion_windows(self):
        current_velocity = self.ship[-1].velocity
        current_yawrate = self.ship[-1].yawrate
        maxvelocity = np.min([self.maxvelocity, current_velocity + self.maxlinearacc * self.dt])
        minvelocity = np.max([self.minvelocity, current_velocity - self.maxlinearacc * self.dt])
        maxyawrate = current_yawrate + self.maxdyawrate * self.dt
        minyawrate = current_yawrate - self.maxdyawrate * self.dt

        return np.array([minvelocity, maxvelocity, minyawrate, maxyawrate])

    # ------------------------------------
    # 三项成本函数的定义-------------------
    def cost_goal(self, locus):
        return distance(np.array([locus[-1].x, locus[-1].y]), self.goal) * self.to_goal_coeff

    def cost_velocity(self, locus):
        return (self.maxvelocity - locus[-1].velocity) * self.velocity_coeff

    def cost_obstacle(self, locus):

        dis = []
        for i in locus:
            for ii in self.obstacle:
                dis.append(distance(np.array([i.x, i.y]), ii))
        dis_np = np.array(dis)
        return 1.0 / np.min(dis_np)

    def cost_total(self, locus):
        return self.cost_goal(locus) + self.cost_velocity(locus) + self.cost_obstacle(locus)

    # -----------------------------------
    # 遍历动态窗口内所有轨迹，调用成本函数计算成本并且择优计算最优速度与加速度----
    def search_for_best_uv(self):
        windows = self.motion_windows()
        best_uv = np.array([0, 0])
        currentcost = np.inf
        # initship=copy.deepcopy(self.ship)
        initship = self.ship[:]
        best_locus = []

        for i in np.arange(windows[0], windows[1], self.velocityres):
            # beginw=time.time()
            for ii in np.arange(windows[2], windows[3], self.yawrateres):
                locus = []
                # for t in np.arange(0,self.predicttime,self.dt):
                #     locus.append(self.motion(i,ii))
                t = 0
                while (t <= self.predicttime):
                    locus.append(self.motion(i, ii))
                    t = t + self.dt
                newcost = self.cost_total(locus)
                if currentcost > newcost:
                    currentcost = newcost
                    best_uv = [i, ii]
                    # best_locus=copy.deepcopy(locus)
                    best_locus = locus[:]

                self.ship = initship[:]

            self.ship = initship[:]

        self.ship = initship[:]
        self.show_animation(best_locus)
        return best_uv, currentcost

    # -----------------------------------------------------------------
    # 用matplotlib画图，一帧一帧图像交替，展示动态效果--------------------
    def show_animation(self, locus):
        plt.cla()
        plt.scatter(self.obstacle[:, 0], self.obstacle[:, 1], s=5)
        plt.plot(self.goal[0], self.goal[1], "ro")
        x = []
        y = []
        for i in locus:
            x.append(i.x)
            y.append(i.y)
        plt.plot(x, y, "g-")
        plt.plot(self.ship[-1].x, self.ship[-1].y, "ro")
        plt.arrow(self.ship[-1].x, self.ship[-1].y, 2 * math.cos(self.ship[-1].yaw), 2 * math.sin(self.ship[-1].yaw),
                  head_length=1.5 * 0.1 * 6, head_width=0.1 * 4)
        plt.grid(True)
        plt.xlim([-10, self.goal[0] * 1.3])
        plt.ylim([-10, self.goal[1] * 1.3])
        plt.pause(0.0001)

    # -------------------------------------------------------------------
    # 安全检测，判断船离障碍物是不是撞上了，目前还没有想到什么比较好的重启小船的函数，只能让小船休眠
    def safedetect(self):
        positionx = self.ship[-1].x
        positiony = self.ship[-1].y
        position = np.array([positionx, positiony])
        for i in self.obstacle:
            if distance(position, i) <= self.saferadius:
                print("撞上了！")
                time.sleep(100000)


# --------------------------------------------------------------------
# 障碍物类，实现障碍物随机运动，本来想生成随机数目坐标随机的障碍物，但是想想还是算了---
class obstacle(object):
    def __init__(self):
        self.assemble = 3 * np.array([
            [-1, -1],
            [0, 2],
            [4.0, 2.0],
            [4.0, 1.0],
            [5.0, 4.0],
            [2.5, 4.0],
            [5.0, 5.0],
            [5.0, 2.5],
            [5.0, 6.0],
            [5.0, 9.0],
            [6.0, 6.0],
            [7.0, 6.0],
            [10.0, 8.0],
            [10.0, 4.0],
            [8.0, 9.0],
            [7.0, 9.0],
            [12.0, 12.0]
        ])
        self.locus = np.vstack(([self.assemble],))

    def update(self):
        for i in self.assemble:
            alpha = 2 * np.random.random() * np.pi
            i[0] = i[0] + 0.2 * np.cos(alpha)
            i[1] = i[1] + 0.2 * np.sin(alpha)
        self.locus = np.vstack((self.locus, [self.assemble]))

    def returnlocus(self, index):
        lx = []
        ly = []
        for i in self.locus:
            lx.append(i[index][0])
            ly.append(i[index][1])
        return lx, ly


# -----------------------------------------------------------------
def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


if __name__ == '__main__':
    state1 = state(0, 10, 0, 0.2, math.pi / 2, 0, 0, 0)
    ship = ship()
    obstacle = obstacle()
    # 初始化
    ship.initialState(state1)
    ship.initialgoal(np.array([35, 35]))
    ship.initialobstacle(obstacle)

    cost = 0
    best_uv = np.array([0, 0])
    # 这里的for循环可以改成while循环，在船到达之前都要运动迭代
    for i in range(1000):
        time_begin = time.time()
        best_uv, cost = ship.search_for_best_uv()
        newstate = ship.motion(best_uv[0], best_uv[1])
        ship.ship[-1].cost = cost
        # ship.obstacle_in_windows()
        time_end = time.time()
        obstacle.update()
        ship.initialobstacle(obstacle)
        ship.safedetect()
        print("第%d次迭代，耗时%.6fs，当前距离终点%.6f" % (
        i + 1, (time_end - time_begin), distance(np.array([ship.ship[-1].x, ship.ship[-1].y]), ship.goal)))
        if distance(np.array([ship.ship[-1].x, ship.ship[-1].y]), ship.goal) < ship.saferadius:
            print("Done!")
            break
        # print("当前轨迹值：")
        # for i in ship.ship:
        #     print("(%.2f,%.2f)"%(i.x,i.y))
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))

    ax[0].scatter(ship.obstacle[:, 0], ship.obstacle[:, 1], s=5)
    ax[0].plot(ship.goal[0], ship.goal[1], "ro")
    lx = []
    ly = []
    lt = []
    lc = []
    for i in ship.ship:
        lx.append(i.x)
        ly.append(i.y)
        lt.append(i.time)
        lc.append(i.cost)
    ax[0].plot(lx, ly)
    for i in range(17):
        locusx, locusy = obstacle.returnlocus(i)
        ax[0].plot(locusx, locusy)
    ax[0].grid(True)
    ax[0].set_title(label="locus figure")
    ax[1].scatter(lt, lc, s=2)
    ax[1].grid(True)
    ax[1].set_title("cost figure")

    plt.show()