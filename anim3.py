import numpy as np
from p1 import convexHull
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class ParticleBox:
    def __init__(self,
                 init_state = [[1, 0, 0, -1],
                               [-0.5, 0.5, 0.5, 0.5],
                               [-0.5, -0.5, -0.5, 0.5]],
                 bounds = [-2, 2, -2, 2],
                 size = 0.04):
        self.init_state = np.asarray(init_state, dtype=float)
        self.size = size
        self.state = self.init_state.copy()
        self.time_elapsed = 0
        self.bounds = bounds

    def step(self, dt):
        self.time_elapsed += dt
        # update positions
        self.state[:, :2] += dt * self.state[:, 2:]
        # check for crossing boundary
        crossed_x1 = (self.state[:, 0] < self.bounds[0] + self.size)
        crossed_x2 = (self.state[:, 0] > self.bounds[1] - self.size)
        crossed_y1 = (self.state[:, 1] < self.bounds[2] + self.size)
        crossed_y2 = (self.state[:, 1] > self.bounds[3] - self.size)
        self.state[crossed_x1, 0] = self.bounds[0] + self.size
        self.state[crossed_x2, 0] = self.bounds[1] - self.size
        self.state[crossed_y1, 1] = self.bounds[2] + self.size
        self.state[crossed_y2, 1] = self.bounds[3] - self.size
        self.state[crossed_x1 | crossed_x2, 2] *= -1
        self.state[crossed_y1 | crossed_y2, 3] *= -1

#------------------------------------------------------------
# set up initial state
np.random.seed(0)
init_state = -0.5 + np.random.random((10, 4))
init_state[:, :2] *= 3.9

box = ParticleBox(init_state, size=0.04)
dt = 1. / 30 # 30fps
#------------------------------------------------------------
# set up figure and animation
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax=fig.add_subplot(111)
ax.set_xlim([-3.2, 3.2])
ax.set_ylim([-2.4, 2.4])
particles, = ax.plot([], [], 'bo', ms=6)
hull, = ax.plot([], [], 'b-', ms=3)

# rect is the box edge
rect = plt.Rectangle(box.bounds[::2],
                     box.bounds[1] - box.bounds[0],
                     box.bounds[3] - box.bounds[2],
                     ec='none', lw=2, fc='none')
ax.add_patch(rect)

def init():
    """initialize animation"""
    global box, rect
    particles.set_data([], [])
    hull.set_data([], [])
    rect.set_edgecolor('none')
    return particles, hull, rect

def animate(i):
    """perform animation step"""
    global box, rect, dt, ax, fig
    box.step(dt)
    # ms = int(fig.dpi * 2 * box.size * fig.get_figwidth()/ np.diff(ax.get_xbound())[0])
    # update pieces of the animation
    rect.set_edgecolor('k')
    particles.set_data(box.state[:, 0], box.state[:, 1])
    hull_p=convexHull([(x, y) for x, y in zip(box.state[:, 0], box.state[:, 1])])
    hull_p.append(hull_p[0])
    hull.set_data([p[0] for p in hull_p], [p[1] for p in hull_p])
    particles.set_markersize(box.size*200)
    return particles, hull, rect

ani = animation.FuncAnimation(fig, animate, frames=600, interval=10, blit=True, init_func=init)
plt.show()