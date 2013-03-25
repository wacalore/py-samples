import numpy as np
import matplotlib.pyplot as plt
import time
from math import *
from scipy.stats import norm

perftime = 0.0

def d1(S, K, r, q, sigma, tau):
    return float((log(S/K) + ((r - q) + 0.5*pow(sigma, 2.0)*tau))/(sigma * sqrt(tau)))

def d2(S, K, r, q, sigma, tau):
    d_2 = float((log(S/K) + ((r - q) - 0.5*pow(sigma, 2.0)*tau))/(sigma * sqrt(tau)))
    return d1(S,K,r,q,sigma,tau) - sigma*sqrt(tau)

def BSFEurCall(S, K, r, q, sigma, tau):
    call = S*exp(-q*tau)*norm.cdf(d1(S, K, r, q, sigma, tau)) - K*exp(-r*tau)*norm.cdf(d2(S, K, r, q, sigma, tau))
    return call

def BSFEurPut(S, K, r, q, sigma, tau):
    return norm.cdf(-1.0*d2(S, K, r, q, sigma, tau))*K*exp(-1.0*r*tau) - norm.cdf(-1.0*d1(S,K,r,q,sigma,tau))*S*exp(-1.0*q*tau)

def vega(S, K, r, q, sigma, tau):
    return K*exp(-r*tau)*norm.pdf(d2(S,K,r,q,sigma,tau))*sqrt(tau)

# a - left endpoint of interval
# b - right endpoint of interval
def ivByBisection(functor, price, S, K, r, q, tau, tolerance, start, end):
    a = start
    b = end
    stime = time.time()
    while 1:
        x = a + (b-a)/2
        fx = functor(S,K,r,q,x,tau) - price
        if (b-a)/2 < tolerance:
            root = x
            etime = time.time()
            perftime = etime - stime
            return root
        if fx*(functor(S,K,r,q,a, tau) - price) > 0:
            a = x
        else:
            b = x

def ivBySecant(functor, price, S, K, r,q, tau, tolerance, start, end):
    a = start
    b = end
    stime = time.time()
    while 1:
        diff = (functor(S,K,r,q,b,tau)-price) - (functor(S,K,r,q,a,tau)-price)
        if abs(diff) < tolerance:
            etime = time.time()
            perftime = etime - stime
            return b
        x = b - ((functor(S,K,r,q,b,tau) - price)*(b-a)/diff)
        a = b
        b = x

def ivByNewton(functor, price, S, K, r, q, tau, tolerance, guess):
    x = guess
    stime = time.time()
    while 1:
        fx = functor(S,K,r,q,x,tau)-price
        x1 = x - fx/vega(S,K,r,q,x,tau)
        if abs(x - x1) < tolerance:
            etime = time.time()
            perftime = etime - stime       
            return x1
        x = x1

def ivSpread(functor, price, S, K1, K2, K3, r, q, tau, tolerance, start, end):        
    a = start
    b = end
    while 1:
        x = a + (b-a)/2
        fx = functor(S, K1, K2, K3, r, q, x, tau) - price
        if (b-a)/2 < tolerance:
            return x
        if fx*(functor(S, K1, K2, K3, r, q, a, tau) - price) > 0:
            a = x
        else:
            b = x


def spreadOption(S, K1, K2, K3, r, q, sigma, tau):
    return 3.0*BSFEurCall(S, K2, r, q, sigma, tau) - 3.0*BSFEurPut(S, K1, r, q, sigma, tau) - 3.0*BSFEurCall(S, K3, r, q, sigma, tau)

print("---------------------------\nSwap Rate + Dividend Yield\n---------------------------")
print("895 Call: %s\n950 Call: %s\n" %( ivByBisection(BSFEurCall, 61.55, 892.0, 895.0, .0033, .015, 359./360., 10e-6, .1, .3), ivByBisection(BSFEurCall, 36.55, 892.0, 950.0, .0033, .015, 359./360., 10e-6, .1, .3)))

print("Bisection: ")
print ivByBisection(BSFEurPut, 77., 892.0, 890.0, .0033, .015, 359./360., 10e-6, .1, .3)
print("Time: %s\n" %(perftime))
print("Secant: ")
print ivBySecant(BSFEurPut, 77., 892.0, 890.0, .0033, .015, 359./360., 10e-6, .1, .3)
print("Time: %s\n" %(perftime))
print("Newton: ")
print ivByNewton(BSFEurPut, 77., 892.0, 890.0, .0033, .015, 359./360., 10e-6, .2)
print("Time: %s]n" %(perftime))

print("---------------------------\nSwap Rate No Dividend\n---------------------------")
print("895 Call: %s\n950 Call: %s\n" %( ivByBisection(BSFEurCall, 61.55, 892.0, 895.0, .0033, 0., 359./360., 10e-6, .1, .3), ivByBisection(BSFEurCall, 36.55, 892.0, 950.0, .0033, 0., 359./360., 10e-6, .1, .3)))
print("Bisection: ")
print ivByBisection(BSFEurPut, 77., 892.0, 890.0, .0033, 0., 359./360., 10e-6, .1, .3)
print("Secant: ")
print ivBySecant(BSFEurPut, 77., 892.0, 890.0, .0033, 0., 359./360., 10e-6, .1, .3)
print("Newton: ")
print ivByNewton(BSFEurPut, 77., 892.0, 890.0, .0033, 0., 359./360., 10e-6, .2)

print("---------------------------\nTreasury Rate No Dividend\n---------------------------")
print("895 Call: %s\n950 Call: %s\n" %( ivByBisection(BSFEurCall, 61.55, 892.0, 895.0, .0014, 0., 359./360., 10e-6, .1, .3), ivByBisection(BSFEurCall, 36.55, 892.0, 950.0, .0014, 0., 359./360., 10e-6, .1, .3)))
print("Bisection: ")
print ivByBisection(BSFEurPut, 77., 892.0, 890.0, .0014, 0., 359./360., 10e-6, .1, .3)
print("Secant: ")
print ivBySecant(BSFEurPut, 77., 892.0, 890.0, .0014, 0., 359./360., 10e-6, .1, .3)
print("Newton: ")
print ivByNewton(BSFEurPut, 77., 892.0, 890.0, .0014, 0., 359./360., 10e-6, .2)

print("\n")
avgPrice = 3.0*np.mean([59.9, 63.2]) - 3.0*np.mean([75.3, 78.7]) - 3.0*np.mean([35.0, 38.1])
bidaskPrice = 3.0*59.9 - 3.0*78.7 - 3.0*38.1
print("Implied Volatility for Average Price: ")
print ivSpread(spreadOption, avgPrice, 892.0, 890.0, 895.0, 950.0, .0033, 0.015, 359./360., 10e-6, .1, .3) 
print("\nImplied Volatility for Bid - Ask Prices: ")
vol = ivSpread(spreadOption, bidaskPrice, 892.0, 890.0, 895.0, 950.0, .0033, 0.015, 359./360., 10e-6, .1, .3)
print(vol)

expiration_90_1vol = list()
expiration_90_75vol = list()
expiration_90_150vol = list()
expiration_180_1vol = list()
expiration_180_75vol = list()
expiration_180_150vol = list()
payoff = list()

for i in range(800, 1000, 2):
    expiration_90_1vol.append(spreadOption(float(i), 890.0, 895.0, 950.0, .0033, .015, vol, 90.0/365.0))
    expiration_90_75vol.append(spreadOption(float(i), 890.0, 895.0, 950.0, .0033, .015, .75*vol, 90.0/365.0))
    expiration_90_150vol.append(spreadOption(float(i), 890.0, 895.0, 950.0, .0033, .015, 1.5*vol, 90.0/365.0))
    expiration_180_1vol.append(spreadOption(float(i), 890.0, 895.0, 950.0, .0033, .015, vol, 180.0/365.0))
    expiration_180_75vol.append(spreadOption(float(i), 890.0, 895.0, 950.0, .0033, .015, .75*vol, 180.0/365.0))
    expiration_180_150vol.append(spreadOption(float(i), 890.0, 895.0, 950.0, .0033, .015, 1.5*vol, 180.0/365.0))
    payoff.append( 3.0*(max(i-895.0, 0.0)) - 3.0*(max(890.0-i, 0.0)) - 3.0*(max(0.0, i - 950)))

x = range(800,1000, 2)
plt.clf()
plt.plot(x, np.array(expiration_90_1vol),label='1.0*vol')
plt.plot(x, np.array(expiration_90_75vol),label='.75*vol')
plt.plot(x, np.array(expiration_90_150vol), label='1.5*vol')
plt.plot(x, np.array(payoff), label='payoff')
plt.grid(True)
plt.title('90 Days to Expiration')
plt.xlabel('Spot Price')
plt.ylabel('Black-Scholes Spread Price')
plt.legend(loc=4)
title = "90days.png"

plt.savefig(title, format='png')

plt.clf()
plt.plot(x, np.array(expiration_180_1vol),label='1.0*vol')
plt.plot(x, np.array(expiration_180_75vol),label='.75*vol')
plt.plot(x, np.array(expiration_180_150vol), label='1.5*vol')
plt.plot(x, np.array(payoff), label='payoff')
plt.grid(True)
plt.title('180 Days to Expiration')
plt.xlabel('Spot Price')
plt.ylabel('Black-Scholes Spread Price')
plt.legend(loc=4)
title = "180days.png"
plt.savefig(title, format='png')
