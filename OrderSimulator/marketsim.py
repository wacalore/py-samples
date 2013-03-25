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

def ExtractSyms(rawlist):
    unique_syms=[]
    for sym in rawlist:
        if sym not in unique_syms:
            unique_syms.append(sym)
    return unique_syms

def OrderDays(rawlist):
    unique_days=[]
    for year,month,day in rawlist:
        if [int(year),int(month),int(day)] not in unique_days:
            unique_days.append([int(year),int(month),int(day)])
    return sorted(sorted(sorted(unique_days, key=itemgetter(2)), key=itemgetter(1)), key=itemgetter(0))

if len(sys.argv) == 4:
    order_file = str(sys.argv[2]).strip()
    cash = float(sys.argv[1])
    values = str(sys.argv[3])
    stock_portfolio = 0
else:
    print "Insufficient arguments"
    exit()

orders = []
data = open(order_file)
ofile = open(values, 'wb')
writer = csv.writer(ofile)

for line in data:
    line = line.rstrip().replace(" ", "")
    order = line.split(',')
    orders.append(order)
   
symbols = ExtractSyms(row[3] for row in orders)
days = OrderDays(row[0:3] for row in orders)
startday = dt.datetime(days[0][0], days[0][1], days[0][2])
endday = dt.datetime(days[len(days)-1][0], days[len(days)-1][1], days[len(days)-1][2]) + dt.timedelta(days=1)
timeofday=dt.timedelta(hours=16)
timestamps = du.getNYSEdays(startday,endday,timeofday)

dataobj = da.DataAccess('Yahoo')
close = dataobj.get_data(timestamps, symbols, "close")

shares  = {}
for sym in symbols:
    shares[sym] = 0

for times in timestamps:
    #date = datetime.datetime(int(year), int(month), int(day), 16, 0)
    stock_portfolio = 0
    for sym in shares:
        stock_portfolio += shares[sym] * close.xs(times)[sym]

    portfolio = cash + stock_portfolio
    writer.writerow([times.year,times.month,times.day,portfolio])

    for row in orders:
        if (int(row[0]) == int(times.year) and int(row[1]) == int(times.month) and int(row[2]) == int(times.day)):
           sym = str(row[3])
           price = close.xs(it.ifilter(lambda x: x >= times, timestamps).next())[sym]
           if str(row[4]).lower() == "buy":
               shares[sym] += int(row[5])
               cash -= price * float(row[5])
           elif row[4].lower() == "sell": 
               shares[sym] -= int(row[5])
               cash += price * float(row[5])
           else:
               continue

data.close
ofile.close

#if __name__ == '__main__':
#    main()

