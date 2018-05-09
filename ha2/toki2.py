import math
import random
from math import pi
import matplotlib.pyplot as plt
import itertools
import numpy


class PolarPoint:
    def __init__(self, radius=0.0, azimuth=0.0):
        self.radius = radius
        self.azimuth = azimuth if azimuth < 2 * math.pi else azimuth - (int(azimuth / (2 * math.pi))) * 2 * math.pi

    def __key(self):
        return (self.radius, self.azimuth)

    def __eq__(x, y):
        return x.__key() == y.__key()

    def __hash__(self):
        return hash(self.__key())

    def hyperbolic_distance(self, p2):
        dtheta = math.pi - math.fabs(math.pi - math.fabs(self.azimuth - p2.azimuth))
        return self.radius + p2.radius + 2 * math.log(dtheta / 2)

    def to_cartesian(self):
        return (self.radius * math.cos(self.azimuth), self.radius * math.sin(self.azimuth))


def test_polar_point_creation():
    p1 = PolarPoint(1, pi / 4)
    assert p1.azimuth == pi / 4
    assert p1.radius == 1


def test_polar_point_creation_with_greater_azimuths():
    p1 = PolarPoint(1, 5 * pi + pi / 4)
    assert math.isclose(p1.azimuth, pi + pi / 4)
    assert p1.radius == 1


def test_hyperbolic_distance():
    p1 = PolarPoint(1, pi / 4)
    p2 = PolarPoint(1, 5 * pi + pi / 4)
    assert not math.isclose(0, p1.hyperbolic_distance(p2))


def lte(a, b):
    return math.isclose(a, b) or a < b


class Graph:
    def __init__(self, nodecount, radius, distribution=lambda x: math.exp(x)):
        self.nodecount = nodecount
        self.radius = radius
        self.points = self.generate_points(distribution)
        self.links = self.generate_links()

    def generate_links(self):
        return [p for p in itertools.combinations(self.points, 2) if lte(p[0].hyperbolic_distance(p[1]), self.radius)]

        # according to Eq.8

    def generate_points(self, distribution=lambda x: math.exp(x)):
        points = []
        for i in range(0, self.nodecount):
            azimuth = random.uniform(0, 2 * math.pi)
            d_point = (random.uniform(0, self.radius), random.uniform(0, distribution(self.radius)))
            while True:
                d_accept = distribution(d_point[0])
                if lte(d_point[1], d_accept):
                    break
                d_point = (random.uniform(0, self.radius), random.uniform(0, distribution(self.radius)))
            points.append(PolarPoint(radius=d_point[0], azimuth=azimuth))
        return points

    def plot(self):
        r = [p.radius for p in self.points]
        theta = [p.azimuth for p in self.points]
        ax = plt.subplot(projection='polar')
        ax.set_rmax(self.radius)
        ax.spines['polar'].set_visible(False)
        ax.scatter(theta, r, c=theta, zorder=3)
        ax.set_rticks([self.radius / 4, self.radius / 2, self.radius * 3 / 4
                          , self.radius])
        ax.set_thetagrids([])
        for l in self.links:
            ax.plot((l[0].azimuth, l[1].azimuth), (l[0].radius, l[1].radius), color='black', marker="None", zorder=0,
                    linewidth=0.25)
        plt.show(ax)


nodes = 500
radius = 14
g = Graph(nodes, radius)
g.plot()
a = 0
