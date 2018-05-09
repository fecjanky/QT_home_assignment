import math
import random
from math import pi
import matplotlib.pyplot as plt
import itertools
import time
import argparse
import networkx as nx
import statistics
import collections


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
        return self.radius + p2.radius + 2 * math.log(math.sin(dtheta / 2))

    def to_cartesian(self):
        return self.radius * math.cos(self.azimuth), self.radius * math.sin(self.azimuth)


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

    # according to Eq.8
    def generate_links(self):
        return [p for p in itertools.combinations(self.points, 2) if lte(p[0].hyperbolic_distance(p[1]), self.radius)]

    # use rejection sampling to generate points with a given distribution
    def generate_points(self, distribution=None):
        points = []
        if distribution is None:
            distribution = lambda r: math.sinh(r) / (math.cosh(self.radius) - 1)

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
        plt.clf()
        plt.close()

    def as_nx_graph(self):
        G = nx.Graph()
        G.add_nodes_from(self.points)
        G.add_edges_from(self.links)
        return G

    def get_avg_degs(self):
        def nodes_closer_than(radius):
            return (n for n in self.points if lte(n.radius, radius))

        G = self.as_nx_graph()
        degs = ((r, dict(G.degree(nodes_closer_than(r))).values()) for r in range(1, self.radius + 1, 1))
        avg_degs = ((r, statistics.mean(a)) for r, a in degs if len(a) > 0)
        return zip(*avg_degs)

    def plot_degree_vs_radius(self, filename=None):
        radius, avg_degs = self.get_avg_degs()
        r = range(0, radius[-1] + 1, 1)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.semilogy(radius, avg_degs, color='black', marker='x', label="simulated avg. degree")
        ax.semilogy(r, list(map(lambda r: 4 / math.pi * self.nodecount * math.exp(-r / 2), r)), color='red',
                    label='theoretical avg. degree')
        ax.set_xlabel('radius')
        ax.set_ylabel('average degree')
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                  ncol=2, mode="expand", borderaxespad=0.)
        outfile = filename if filename is not None else 'graph_stats.png'
        fig.savefig(outfile, orientation='landscape', dpi=1200)
        fig.clf()
        plt.close()

    def plot_stats(self, filename=None):
        radius, avg_degs = self.get_avg_degs()
        print("Average degree of network radius={radius} and nodecount={nodes} is {avg}".format(radius=self.radius,
                                                                                                nodes=self.nodecount,
                                                                                                avg=avg_degs[-1]))
        degree_sequence = sorted([d for n, d in self.as_nx_graph().degree()], reverse=True)
        degreeCount = collections.Counter(degree_sequence)
        deg, cnt = zip(*degreeCount.items())
        scnt = sum(cnt)
        cnt = list(map(lambda x: x / scnt, cnt))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.loglog(deg, cnt, marker='o', color="blue", linestyle='None', label="empirical distribution")
        ax.set_ylabel("P(k)")
        ax.set_xlabel("Node degree k")
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=4,
                  ncol=2, mode="expand", borderaxespad=0.)
        outfile = filename if filename is not None else 'graph_stats.png'
        fig.savefig(outfile, orientation='landscape', dpi=1200)
        fig.clf()
        plt.close()


parser = argparse.ArgumentParser(description='hyperbolic geometry complex network analysis',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-p', '--plot', help='plot to file, default name is graph.png', action='store_true')
parser.add_argument('-s', '--stats', help='print out stats', action='store_true')
parser.add_argument('-n', '--nodes', type=int, default=100, help='number of nodes to use in the generator')
parser.add_argument('-r', '--radius', type=int, default=14, help='the radius of the disc to be used')

args = parser.parse_args()
nodes = args.nodes
radius = args.radius

start_time = time.time()

g = Graph(nodes, radius)
print(len(g.points))
if args.plot:
    g.plot()
if args.stats:
    g.plot_stats()

print("Execution took {value} seconds".format(value=(time.time() - start_time)))
