import time
import itertools
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
    ax.set_xlabel('time')
    ax.set_ylabel('intensity')
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


def plotpacketcountcorrelation(packetcount):
    fig, ax = plt.subplots()
    y_packetcountcorrelation = []
    for lag in range(0, 500):
        y_packetcountcorrelation.append(autocorr(packetcount, lag)[0, 1])
    x_packetcountcorrelation = np.linspace(0, 500, 500)
    plt.plot(x_packetcountcorrelation, y_packetcountcorrelation)
    ax.set_xlabel('lag')
    ax.set_ylabel('correlation')
    fig.savefig('packetcountcorrelation.png', orientation='landscape', dpi=600)


def plotidi(interarr, k):
    fig, ax = plt.subplots()
    idilist = []
    for m in range(1, k):
        idilist.append(idi(interarr, m))
    idi_x = np.linspace(0, k, k - 1)
    plt.plot(idi_x, idilist)
    ax.set_xlabel('lag')
    ax.set_ylabel('IDI')
    fig.savefig('idi.png', orientation='landscape', dpi=600)


def plotidc(counts, t):
    fig, ax = plt.subplots()
    idclist = []
    for m in range(1, t):
        idclist.append(idi(counts, m))
    idc_x = np.linspace(0, t, t - 1)
    plt.plot(idc_x, idclist)
    ax.set_xlabel('time')
    ax.set_ylabel('IDC')
    fig.savefig('idc.png', orientation='landscape', dpi=600)


###########################################
# command handling
###########################################

class commandeExecutor:
    def __init__(self, args):
        self.timestamps = parsefile(args.input)
        self.interarrivals = np.fromiter(getinterarrivals(self.timestamps), np.float64, len(self.timestamps) - 1)
        self.packetcounter = getintensity(self.timestamps)

    def execute(self, command):
        if not hasattr(self, "command_" + command):
            raise AttributeError("unknown command");
        getattr(self, "command_" + command)()

    def command_interarrival(self):
        plotinterarrivalpdf(self.interarrivals)

    def command_intensity(self):
        plotintensity(self.packetcounter)

    def command_packetcountcorrelation(self):
        plotpacketcountcorrelation(self.packetcounter)

    def command_arrtimecorrelation(self):
        plotarrtimecorrelation(self.interarrivals)

    def command_idi(self):
        plotidi(self.interarrivals, 50)

    def command_idc(self):
        plotidc(self.packetcounter, 200)

    def command_stats(self):
        print("PMR: {value}".format(value=peaktomean(self.interarrivals)))
        print("SCV: {value}".format(value=scv(self.interarrivals)))
        print("Third moment: {value}".format(value=thirdmoment(self.interarrivals)))

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
        help += c.replace("command_", "") + "\n"
    help += "\n"
    return help


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Traffic statistics.', epilog=listcommands())

    parser.add_argument('command', type=str,
                        help='comand to be executed', nargs='+')
    parser.add_argument('-i', '--input', type=str,
                        help='filename')

    start_time = time.time()

    args = parser.parse_args()
    executor = commandeExecutor(args)
    for c in args.command:
        executor.execute(c)
    print("Execution took {value} seconds".format(value=(time.time() - start_time)))
