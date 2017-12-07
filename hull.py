from collections import namedtuple
from operator import itemgetter
from math import sqrt, hypot
from functools import cmp_to_key

Point = namedtuple('Point', ['x', 'y'])

def dist(p, q):
    return hypot(p.x - q.x, p.y - q.y)


def orientation(p, q, r):
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    if val == 0:
        return 0
    return 1 if val > 0 else 2


def compare(p, q):
    o = orientation(p0, p, q)
    if o == 0:
        return -1 if dist(p0, q) >= dist(p0, p) else 1
    return -1 if o == 2 else 1


def convexHull(points):
    points = [Point._make(x) for x in points]
    hull = []
    global p0
    p0 = min(points, key=itemgetter(1, 0))
    hull.append(p0)
    points.remove(p0)
    points.sort(key=cmp_to_key(compare))
    i = 0
    while(i < len(points)):
        while((i < len(points) - 1) and (orientation(p0, points[i], points[i + 1]) == 0)):
            i += 1
        hull.append(points[i])
        i += 1
    if len(hull) < 3:
        return []
    res = hull[0:3]
    for i in range(3, len(hull)):
        while(orientation(res[-2], res[-1], hull[i]) != 2):
            res.pop()
        res.append(hull[i])
    return res