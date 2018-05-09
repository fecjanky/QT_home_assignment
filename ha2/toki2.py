import math
import random
from math import pi
import matplotlib.pyplot as plt
import itertools
import time
import argparse


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

    def plot(self, filename=None):
        r = [p.radius for p in self.points]
        theta = [p.azimuth for p in self.points]
        plt.clf()
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')

        ax.set_rmax(self.radius)
        ax.spines['polar'].set_visible(False)
        ax.scatter(theta, r, c=theta, s=2, zorder=3)
        ax.set_rticks([self.radius / 4, self.radius / 2, self.radius * 3 / 4
                          , self.radius])
        ax.set_thetagrids([])
        for l in self.links:
            ax.plot((l[0].azimuth, l[1].azimuth), (l[0].radius, l[1].radius), color='black', marker="None", zorder=0,
                    linewidth=0.25)
        outfile = filename if filename is not None else 'graph.png'
        fig.savefig(outfile, orientation='landscape', dpi=1200)

    def print_stats(self):
        print("stats")


parser = argparse.ArgumentParser(description='hyperbolic geometry complex network analysis',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-p', '--plot', type=str,
                    help='plot to file, default name is graph.png', nargs='?')
parser.add_argument('-s', '--stats', action='store_true',
                    help='print out stats')
parser.add_argument('-n', '--nodes', type=int, default=100, help='number of nodes to use in the generator')
parser.add_argument('-r', '--radius', type=int, default=14, help='the radius of the disc to be used')

args = parser.parse_args()
nodes = args.nodes
radius = args.radius

start_time = time.time()

g = Graph(nodes, radius)
if args.plot:
    g.plot(args.plot[0] if len(args.plot) > 0 else None)
if args.stats:
    g.print_stats()

print("Execution took {value} seconds".format(value=(time.time() - start_time)))
