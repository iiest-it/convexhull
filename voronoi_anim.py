import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.animation as animation
from ParticleBox import ParticleBox

init_state = -0.5 + np.random.random((10, 4))
init_state[:, :2] *= 3.9

box = ParticleBox(init_state, size=0.04)
dt = 1. / 30

plt.ion()
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111)
ax.set_xlim([-3.2, 3.2])
ax.set_ylim([-2.4, 2.4])
# particles, = ax.plot([], [], 'bo', ms=6)
rect = plt.Rectangle(box.bounds[::2],
                     box.bounds[1] - box.bounds[0],
                     box.bounds[3] - box.bounds[2],
                     ec='none', lw=2, fc='none')
ax.add_patch(rect)
# plt.draw()

while True:
    plt.clf()
    points = np.array([[x, y]
                       for x, y in zip(box.state[:, 0], box.state[:, 1])])
    vor = Voronoi(points)

    plt.xlim(-3.2, 3.2)
    plt.ylim(-2.4, 2.4)
    plt.plot(points[:, 0], points[:, 1], 'o')
    plt.plot(vor.vertices[:, 0], vor.vertices[:, 1], '*')
    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            plt.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], 'k-')

    center = points.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.any(simplex < 0):
            i = simplex[simplex >= 0][0]
            t = points[pointidx[1]] - points[pointidx[0]]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = points[pointidx].mean(axis=0)
            far_point = vor.vertices[i] + \
                np.sign(np.dot(midpoint - center, n)) * n * 100
            plt.plot([vor.vertices[i, 0], far_point[0]], [
                vor.vertices[i, 1], far_point[1]], 'k--')
    plt.draw()
    box.step(dt)
    plt.pause(0.00001)
