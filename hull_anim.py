import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ParticleBox import ParticleBox
from hull import convexHull

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