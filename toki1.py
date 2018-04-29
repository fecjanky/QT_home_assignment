import time
import itertools
# import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import argparse
from datetime import datetime, timedelta

# Feladat
"""Homework:
paper: http://hsnlab.tmit.bme.hu/%7Emolnar/files/ilk97ext.pdf

Perform a similar traffic analysis study that is in the first paper with an arbitrary chosen time series.
(It can be measured internet traffic from anywhere from the internet, e.g., http://ita.ee.lbl.gov/ or it can be your 
own measured traffic or even other types of time series.)"""

# Time series chosen:
"""
http://ita.ee.lbl.gov/html/contrib/LBL-TCP-3.html
The trace ran from 14:10 to 16:10 on Thursday, January 20, 1994 (times are Pacific Standard Time), capturing 
1.8 million TCP packets (about 0.0002 of these were dropped). The tracing was done on the Ethernet DMZ network over 
which flows all traffic into or out of the Lawrence Berkeley Laboratory, located in Berkeley, California. The raw 
trace was made using tcpdump on a Sun Sparcstation using the BPF kernel packet filter. 
Timestamps have microsecond precision. """


def parse_datetime(datetimestring):
    return datetime.strptime(datetimestring, "%Y.%m.%d %H:%M:%S.%f")


def test_parse_datetime():
    date = parse_datetime("2018.03.18 08:21:33.420")
    assert date.year == 2018
    assert date.month == 3
    assert date.day == 18
    assert date.hour == 8
    assert date.minute == 21
    assert date.second == 33
    assert date.microsecond == 420000
    pass


def parsefile(filename):
    with open(filename) as f:
        timestamps = [parse_datetime(line.strip()) for line in f if line.strip()]
        t0 = min(timestamps)
        nptimestamps = np.fromiter(((t - t0).total_seconds() for t in timestamps), np.float64, len(timestamps))
        return np.sort(nptimestamps)


def getinterarrivals(arrivals):
    if len(arrivals) is 0:
        return [0]
    i = 1
    while i < len(arrivals):
        yield arrivals[i] - arrivals[i - 1]
        i = i + 1


def plotinterarrivalpdf(interarr):
    sns.distplot(interarr, 100)
    plt.xlabel('interarrival length')
    plt.ylabel('pdf')
    plt.show()


def getintensity(timelist):
    n = 0
    packetcounter = [0]
    for t in timelist:
        if t - timelist[0] > n + 1:
            diff = t - timelist[0]
            n += diff
            packetcounter += itertools.repeat(0, int(diff + 1))
        packetcounter[int(n)] += 1
    return packetcounter


def plotintensity(packetcounter):
    plt.plot(range(0, len(packetcounter)), packetcounter)
    plt.xlabel('time')
    plt.ylabel('intensity')
    plt.show()


def autocorr(x, lag):
    # return np.correlate(x[0:len(x)-lag],x[0:len(x)-lag])[0]
    return np.corrcoef(x[0:len(x) - lag], x[lag:len(x)])
    # return np.corrcoef(np.array([x[0:len(x) - lag], x[lag:len(x)]]))


def plotarrtimecorrelation(interarrivals):
    y_arrtimecorrelation = []
    for lag in range(0, 500):
        y_arrtimecorrelation.append(autocorr(interarrivals, lag)[0, 1])
    x_arrtimecorrelation = np.linspace(0, 500, 500)
    plt.plot(x_arrtimecorrelation, y_arrtimecorrelation)
    plt.xlabel('lag')
    plt.ylabel('correlation')
    plt.show()


def plotpacketcountcorrelation(packetcount):
    y_packetcountcorrelation = []
    for lag in range(0, 500):
        y_packetcountcorrelation.append(autocorr(packetcount, lag)[0, 1])
    x_packetcountcorrelation = np.linspace(0, 500, 500)
    plt.plot(x_packetcountcorrelation, y_packetcountcorrelation)
    plt.xlabel('lag')
    plt.ylabel('correlation')
    plt.show()


# General statistics
# --------------------
"""def mean(arr):
    summa = 0
    for i in arr:
        summa += i
    mean = summa / len(arr)
    return mean"""


# peak to mean ratio
def peaktomean(interarr):
    peak = max(interarr)
    return peak / np.mean(interarr)


# squared coefficient variation
def scv(interarr):
    return np.var(interarr) / (np.mean(interarr) * np.mean(interarr))


# skewness
def thirdmoment(interarr):
    interarray = np.array(interarr)
    return stats.skew(interarray)


# i is arbitrary
def idi(interarr, k):
    sumlist = []
    for i in range(0, len(interarr) - k):
        summa = 0
        for j in range(i, i + k):
            summa += interarr[j]
        sumlist.append(summa)
    variance = np.var(sumlist)
    idival = variance / (k * np.mean(interarr) * np.mean(interarr))
    return idival


def plotidi(interarr, k):
    idilist = []
    for m in range(1, k):
        idilist.append(idi(interarr, m))
    idi_x = np.linspace(0, k, k - 1)
    plt.plot(idi_x, idilist)
    plt.xlabel('lag')
    plt.ylabel('IDI')
    plt.show()


def idc(counts, t):
    sumlist = []
    for i in range(0, len(counts) - t):
        summa = 0
        for j in range(i, i + k):
            summa += counts[j]
        sumlist.append(summa)
    variance = np.var(sumlist)
    idcval = variance / (t * np.mean(counts))
    return idcval


def plotidc(counts, t):
    idclist = []
    for m in range(1, t):
        idclist.append(idi(counts, m))
    idc_x = np.linspace(0, t, t - 1)
    plt.plot(idc_x, idclist)
    plt.xlabel('time')
    plt.ylabel('IDC')
    plt.show()


parser = argparse.ArgumentParser(description='Traffic statistics.')
parser.add_argument('command', type=str,
                    help='comand to be executed')
parser.add_argument('filename', type=str,
                    help='filename')


class commandeExecutor:
    def __init__(self, args):
        self.timestamps = parsefile(args.filename)
        self.interarrivals = np.fromiter(getinterarrivals(self.timestamps), np.float64, len(self.timestamps) - 1)
        self.packetcounter = getintensity(self.timestamps)

    def execute(self, command):
        if not hasattr(self, command):
            raise AttributeError("unknown command");
        getattr(self,command)()

    def interarrival(self):
        plotinterarrivalpdf(self.interarrivals)

    def intensity(self):
        plotintensity(self.packetcounter)

    def packetcountcorrelation(self):
        plotpacketcountcorrelation(self.packetcounter)

    def arrtimecorrelation(self):
        plotarrtimecorrelation(self.interarrivals)

    def idi(self):
        plotidi(self.interarrivals, 50)

    def idc(self):
        plotidc(self.packetcounter, 200)

    def getInterArrivals(self):
        return self.interarrivals

    def getTimeStamps(self):
        return self.timestamps

    def getPacketCounter(self):
        return self.packetcounter

    def stats(self):
        print("PMR: {value}".format(value=peaktomean(self.interarrivals)))
        print("SCV: {value}".format(value=scv(self.interarrivals)))
        print("Third moment: {value}".format(value=thirdmoment(self.interarrivals)))
        print("Execution took {value} seconds".format(value=(time.time() - start_time)))


if __name__ == '__main__':
    start_time = time.time()

    args = parser.parse_args()
    executor = commandeExecutor(args)
    executor.execute(args.command)

