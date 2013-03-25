import csv,sys
import itertools as it
import qstkutil.qsdateutil as du
import qstkutil.tsutil as tsu
import qstkutil.DataAccess as da
import datetime as dt
import numpy as np
import scipy.stats as stpy
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
from operator import itemgetter

startday = dt.datetime(2008,1,1)
endday = dt.datetime(2011,12,31)
timeofday=dt.timedelta(hours=16)
timestamps = du.getNYSEdays(startday,endday,timeofday)

#def main(argv):

benchmark = []
if len(sys.argv) == 3:
    values_file = str(sys.argv[1]).strip()
  #  benchmark = benchmark.append(str(sys.argv[2]).strip())
    benchmark.append(str(sys.argv[2]))
else:
    print "Insufficient arguments"
    exit()


values = []
times = []
data = open(values_file)

for line in data:
    line = line.rstrip().replace(" ", "")
    value = line.split(',')
    times.append(value[0:3])
    values.append(float(value[3]))

startday = dt.datetime(int(times[0][0]), int(times[0][1]), int(times[0][2]))
endday = dt.datetime(int(times[len(times)-1][0]), int(times[len(times)-1][1]), int(times[len(times)-1][2])) + dt.timedelta(days=1)
timeofday = dt.timedelta(hours=16)
timestamps = du.getNYSEdays(startday,endday,timeofday)

#endday = dt.datetime(int
dataobj = da.DataAccess('Yahoo')
close = dataobj.get_data(timestamps, benchmark, "close")
close = pd.DataFrame.fillna(close, method='ffill')
close = pd.DataFrame.fillna(close, method='backfill')
factor = values[0] / close.values[0]
pricedata = np.array([factor * y for y in close.values])
valuesdata = np.array(values)

bench_returns = (pricedata[1:,:]/pricedata[0:-1,:]) - 1 
port_returns = (valuesdata[1:]/valuesdata[0:-1]) - 1
bench_treturn = pricedata[len(pricedata)-1]/pricedata[0]
port_treturn = valuesdata[len(valuesdata)-1]/valuesdata[0]
sd_bench_returns = np.std(bench_returns)
sd_port_returns = np.std(port_returns)
bench_sharpe = sqrt(len(timestamps))* np.mean(bench_returns)/sd_bench_returns  
port_sharpe = sqrt(len(timestamps)) * np.mean(port_returns)/sd_port_returns

print("\n-------------\nBenchmark: %s\nTotal Return: %s\nReturn SD: %s\nBench Sharpe Ratio: %s" %(benchmark, bench_treturn, sd_bench_returns, bench_sharpe))
print("\nPortfolio Total Return: %s\nPortfolio Return SD: %s\nPortfolio Sharpe Ratio: %s (normalized for %s trading days)" %(port_treturn, sd_port_returns, port_sharpe, len(timestamps)))

values_frame = pd.DataFrame(np.array(values), index=close.index)
mod_close = pd.DataFrame(np.array(pricedata), index=close.index)

plt.figure(1)                # the first figure
xlim(startday, endday)
plt.plot(close.index, mod_close, label=benchmark[0])
plt.plot(close.index, values_frame, color='red', label="Portfolio")
plt.legend(loc=3)
savefig('chart.pdf', format='pdf')
