import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import argparse

#Feladat
"""Homework:
paper: http://hsnlab.tmit.bme.hu/%7Emolnar/files/ilk97ext.pdf

Perform a similar traffic analysis study that is in the first paper with an arbitrary chosen time series.
(It can be measured internet traffic from anywhere from the internet, e.g., http://ita.ee.lbl.gov/ or it can be your 
own measured traffic or even other types of time series.)"""

#Time series chosen:
"""
http://ita.ee.lbl.gov/html/contrib/LBL-TCP-3.html
The trace ran from 14:10 to 16:10 on Thursday, January 20, 1994 (times are Pacific Standard Time), capturing 
1.8 million TCP packets (about 0.0002 of these were dropped). The tracing was done on the Ethernet DMZ network over 
which flows all traffic into or out of the Lawrence Berkeley Laboratory, located in Berkeley, California. The raw 
trace was made using tcpdump on a Sun Sparcstation using the BPF kernel packet filter. 
Timestamps have microsecond precision. """


def parsefile(filename):
    with open(filename) as f:
        for line in f:
            lines = line.rstrip('\n')
            columns = lines.split()
            timestamps.append(float(columns[0]))
            data.append(float(columns[5]))


def getinterarrivals(arrivals):
    last = 0
    for x in arrivals:
        previous = last
        last = x
        if (previous != 0):
            interarrivals.append(last - previous)


def plotinterarrivalpdf(interarr):
    sns.distplot(interarr,100)
    plt.xlabel('interarrival length')
    plt.ylabel('pdf')
    plt.show()


def plotintensity(timearray):
    n = 1
    for t in timearray:
        packetcounter[n-1] += 1
        if (t > n):
            n += 1
            packetcounter.append(0)
    x_intensity = np.linspace(0,n,7200)
    plt.plot(x_intensity,packetcounter)
    plt.xlabel('time')
    plt.ylabel('intensity')
    plt.show()


def autocorr(x, lag):
    return np.corrcoef(np.array([x[0:len(x)-lag],x[lag:len(x)]]))


def plotarrtimecorrelation(interarrivals):
    y_arrtimecorrelation = []
    for lag in range (0,500):
        acorr = autocorr(interarrivals,lag)
        y_arrtimecorrelation.append(acorr[0,1])
    x_arrtimecorrelation = np.linspace(0,500,500)
    plt.plot(x_arrtimecorrelation,y_arrtimecorrelation)
    plt.xlabel('lag')
    plt.ylabel('correlation')
    plt.show()


def plotpacketcountcorrelation(packetcount):
    y_packetcountcorrelation = []
    for lag in range (0,500):
        acorr = autocorr(packetcount, lag)
        y_packetcountcorrelation.append(acorr[0,1])
    x_packetcountcorrelation = np.linspace(0,500,500)
    plt.plot(x_packetcountcorrelation,y_packetcountcorrelation)
    plt.xlabel('lag')
    plt.ylabel('correlation')
    plt.show()


#General statistics
#--------------------
"""def mean(arr):
    summa = 0
    for i in arr:
        summa += i
    mean = summa / len(arr)
    return mean"""


#peak to mean ratio
def peaktomean(interarr):
    peak = max(interarr)
    return peak/np.mean(interarr)


#squared coefficient variation
def scv(interarr):
    return np.var(interarr)/(np.mean(interarr) * np.mean(interarr))


#skewness
def thirdmoment(interarr):
    interarray = np.array(interarr)
    return stats.skew(interarray)


#i is arbitrary
def idi(interarr,k):
    sumlist = []
    for i in range(0,len(interarr)-k):
        summa = 0
        for j in range(i,i+k):
            summa += interarr[j]
        sumlist.append(summa)
    variance = np.var(sumlist)
    idival = variance/(k * np.mean(interarr) * np.mean(interarr))
    return idival


def plotidi(interarr,k):
    idilist = []
    for m in range(1,k):
        idilist.append(idi(interarr,m))
    idi_x = np.linspace(0,k,k-1)
    plt.plot(idi_x,idilist)
    plt.xlabel('lag')
    plt.ylabel('IDI')
    plt.show()


def idc(counts,t):
    sumlist = []
    for i in range(0, len(counts) - t):
        summa = 0
        for j in range(i, i + k):
            summa += counts[j]
        sumlist.append(summa)
    variance = np.var(sumlist)
    idcval = variance / (t * np.mean(counts))
    return idcval

def plotidc(counts,t):
    idclist = []
    for m in range(1, t):
        idclist.append(idi(counts, m))
    idc_x = np.linspace(0, t, t - 1)
    plt.plot(idc_x, idclist)
    plt.xlabel('time')
    plt.ylabel('IDC')
    plt.show()

import argparse

parser = argparse.ArgumentParser(description='Traffic statistics.')
parser.add_argument('filename', metavar='N', type=str, nargs=1,
                    help='filename')




if __name__=='__main__':
    start_time = time.time()
    timestamps = []
    data = []
    interarrivals = []
    packetcounter = [0]
    args = parser.parse_args()
    parsefile(args.filename)
    getinterarrivals(timestamps)
    #plotinterarrivalpdf(interarrivals)
    plotintensity(timestamps)
    #plotpacketcountcorrelation(packetcounter)
    #plotarrtimecorrelation(interarrivals)
    #plotidi(interarrivals,50)
    plotidc(packetcounter,200)
    #print "PMR: %s" % peaktomean(interarrivals)
    #print "SCV: %s" % scv(interarrivals)
    #print "Third moment: %s" % thirdmoment(interarrivals)
    print "Execution took %s seconds" % (time.time() - start_time)