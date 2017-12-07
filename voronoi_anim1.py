import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from ParticleBox import ParticleBox

init_state = -0.5 + np.random.random((10, 4))
init_state[:, :2] *= 3.9

box = ParticleBox(init_state, size=0.04)
dt = 1. / 30
points = np.array([[x, y] for x, y in zip(box.state[:, 0], box.state[:, 1])])
vor = Voronoi(points)

fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111)


def init():
    global ax_points, ax_vertices, lc1, lc2
    ax.set_xlim([-3.2, 3.2])
    ax.set_ylim([-2.4, 2.4])
    ax_points, = ax.plot(vor.points[:, 0], vor.points[:, 1], 'b.', ms=12)
    ax_vertices, = ax.plot(vor.vertices[:, 0], vor.vertices[:, 1], 'ro')
    lc1 = LineCollection([], colors='k', lw=1.0, linestyle='solid')
    lc2 = LineCollection([], colors='k', lw=1.0, linestyle='dashed')
    ax.add_collection(lc1)
    ax.add_collection(lc2)
    return ax_points, ax_vertices, lc1, lc2,


def animate(i):
    points = np.array([[x, y]
                       for x, y in zip(box.state[:, 0], box.state[:, 1])])
    vor = Voronoi(points)
    box.step(dt)
    ax_points.set_data(vor.points[:, 0], vor.points[:, 1])
    ax_vertices.set_data(vor.vertices[:, 0], vor.vertices[:, 1])

    line_segments = []
    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            line_segments.append([(x, y) for x, y in vor.vertices[simplex]])

    lc1.set_segments(line_segments)
    lc1.set_alpha(1.0)
    ptp_bound = vor.points.ptp(axis=0)

    line_segments = []
    center = vor.points.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.any(simplex < 0):
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[i] + direction * ptp_bound.max()

            line_segments.append([(vor.vertices[i, 0], vor.vertices[i, 1]),
                                  (far_point[0], far_point[1])])

    lc2.set_segments(line_segments)
    lc2.set_alpha(1.0)
    return ax_points, ax_vertices, lc1, lc2,


ani = animation.FuncAnimation(
    fig, animate, frames=600, interval=10, blit=True, init_func=init)
plt.show()
