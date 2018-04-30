# coding=utf8
import time
import itertools
import math
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import argparse
from datetime import datetime, timedelta


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


def parse_datetime_alt(datetimestring, col=0):
    return float(datetimestring.split()[col])


def test_parse_datetime_alt():
    date = parse_datetime_alt("0.023775 1 2 23 2436 2")
    assert date.second == 0.023775


def parsefile(filename, alternativeSyntax=False, column=0):
    with open(filename) as f:
        if not alternativeSyntax:
            timestamps = [parse_datetime(line.strip()) for line in f if line.strip()]
            t0 = min(timestamps)
            nptimestamps = np.fromiter(((t - t0).total_seconds() for t in timestamps), np.float64, len(timestamps))
            return np.sort(nptimestamps)
        else:
            return np.fromiter((parse_datetime_alt(line.strip(), column) for line in f if line.strip()),
                               np.float64)


def getinterarrivals(arrivals):
    if len(arrivals) is 0:
        return [0]
    return np.subtract(np.array(arrivals[1:]), arrivals[0:-1])


def getintensity(timelist):
    if len(timelist) < 1:
        return []
    return np.histogram(timelist, bins=int(math.floor(timelist[-1])))[0]


def autocorr(x, lag):
    return np.corrcoef(x[0:len(x) - lag], x[lag:len(x)])


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


windowed_sum_cache = {}


def get_windowed_sum(interarr, k):
    assert k > 0
    if (interarr.ctypes.data, k) not in windowed_sum_cache:
        if k == 1:
            windowed_sum_cache[(interarr.ctypes.data, k)] = interarr
        else:
            prev = get_windowed_sum(interarr, k - 1)
            sumlist = np.fromiter((prev[i] + interarr[i + k - 1] for i in range(0, len(prev) - 1)), np.float64)
            windowed_sum_cache[(interarr.ctypes.data, k)] = sumlist
    return windowed_sum_cache[(interarr.ctypes.data, k)]


def idi(interarr, k):
    assert k > 0
    sumlist = get_windowed_sum(interarr, k)
    variance = np.var(sumlist)
    idival = variance / (k * np.mean(interarr) * np.mean(interarr))
    return idival


def idc(counts, t):
    sumlist = get_windowed_sum(counts, t)
    variance = np.var(sumlist)
    idcval = variance / (t * np.mean(counts))
    return idcval


###########################################
# plotting
###########################################

def plotinterarrivalpdf(interarr):
    fig, ax = plt.subplots()
    sns.distplot(interarr, 100)
    ax.set_xlabel('interarrival length[s]')
    ax.set_ylabel('pdf')
    fig.savefig('interarrival.png', orientation='landscape', dpi=600)


def plotintensity(packetcounter):
    fig, ax = plt.subplots()
    plt.plot(range(0, len(packetcounter)), packetcounter)
    ax.set_xlabel('time[s]')
    ax.set_ylabel('intensity[pkt/s]')
    fig.savefig('intensity.png', orientation='landscape', dpi=600)


def plotarrtimecorrelation(interarrivals):
    fig, ax = plt.subplots()
    y_arrtimecorrelation = []
    for lag in range(0, 500):
        y_arrtimecorrelation.append(autocorr(interarrivals, lag)[0, 1])
    x_arrtimecorrelation = np.linspace(0, 500, 500)
    plt.plot(x_arrtimecorrelation, y_arrtimecorrelation)
    ax.set_xlabel('lag')
    ax.set_ylabel('correlation')
    fig.savefig('arrtimecorrelation.png', orientation='landscape', dpi=600)


def plotpacketcountcorrelation(packetcount, lagrange=500):
    fig, ax = plt.subplots()
    y_packetcountcorrelation = []
    for lag in range(0, lagrange):
        y_packetcountcorrelation.append(autocorr(packetcount, lag)[0, 1])
    x_packetcountcorrelation = np.linspace(0, lagrange, lagrange)
    plt.plot(x_packetcountcorrelation, y_packetcountcorrelation)
    ax.set_xlabel('lag')
    ax.set_ylabel('correlation')
    fig.savefig('packetcountcorrelation.png', orientation='landscape', dpi=600)


def plotidi(interarr, k):
    fig, ax = plt.subplots()
    idilist = [idi(interarr, m) for m in range(1, k)]
    idi_x = np.linspace(0, k, k - 1)
    plt.plot(idi_x, idilist)
    ax.set_xlabel('lag')
    ax.set_ylabel('IDI')
    fig.savefig('idi.png', orientation='landscape', dpi=600)


def plotidc(counts, t):
    fig, ax = plt.subplots()
    idclist = [idc(counts, m) for m in range(1, t)]
    idc_x = np.linspace(0, t, t - 1)
    plt.plot(idc_x, idclist)
    ax.set_xlabel('time[s]')
    ax.set_ylabel('IDC')
    fig.savefig('idc.png', orientation='landscape', dpi=600)


###########################################
# command handling
###########################################

class commandeExecutor:
    def __init__(self, args=None):
        if args is not None:
            self.timestamps = parsefile(args.input, args.tcpdump)
            self.interarrivals = np.fromiter(getinterarrivals(self.timestamps), np.float64, len(self.timestamps) - 1)
            self.packetcounter = getintensity(self.timestamps)

    def execute(self, command, args):
        if not hasattr(self, "command_" + command):
            raise AttributeError("unknown command");
        getattr(self, "command_" + command)(args)

    ##########################
    def command_interarrival(self, args):
        plotinterarrivalpdf(self.interarrivals)

    def command_intensity(self, args):
        plotintensity(self.packetcounter)

    def command_packetcountcorrelation(self, sargs):
        parser = argparse.ArgumentParser(description='Packet auto correlation help.', prog="packetcountcorrelation")
        parser.add_argument("--lag", type=int, help="size of lag window", default=500)
        args = parser.parse_args(sargs.split())
        plotpacketcountcorrelation(self.packetcounter, args.lag)

    def command_arrtimecorrelation(self, args):
        plotarrtimecorrelation(self.interarrivals)

    def command_idi(self, args):
        plotidi(self.interarrivals, 50)

    def command_idc(self, args):
        plotidc(self.packetcounter, 200)

    def command_stats(self, args):
        print("PMR: {value}".format(value=peaktomean(self.interarrivals)))
        print("SCV: {value}".format(value=scv(self.interarrivals)))
        print("Third moment: {value}".format(value=thirdmoment(self.interarrivals)))

    ##########################
    def getInterArrivals(self):
        return self.interarrivals

    def getTimeStamps(self):
        return self.timestamps

    def getPacketCounter(self):
        return self.packetcounter


def listcommands():
    commands = [d for d in dir(commandeExecutor) if "command_" in d]
    help = "Supported commands:\n"
    for c in commands:
        help += c.replace("command_", "") + ","
    help += "\n"
    help += " - add 'help' after command to get more help on the command itself"
    return help


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Traffic statistics.', epilog=listcommands())

    parser.add_argument('command', type=str,
                        help='comand to be executed', nargs='+')
    parser.add_argument('-i', '--input', type=str,
                        help='filename')
    parser.add_argument('-a', '--args', type=str,
                        help='args for the command', default="")
    parser.add_argument('-t', '--tcpdump', default=False, action='store_true',
                        help='use tcpdump format for input file', )

    start_time = time.time()

    args = parser.parse_args()

    if "help" in args.command:
        commandeExecutor().execute(args.command[0], args.args + " --help")

    executor = commandeExecutor(args)
    for c in args.command:
        executor.execute(c, args.args)
    print("Execution took {value} seconds".format(value=(time.time() - start_time)))
