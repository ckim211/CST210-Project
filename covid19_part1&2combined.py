#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 13:18:04 2021

This program reads in covid-19 related data, first one being the confirmed case of the virus and the second being the vaccination rate, and fit two plots to a linear model for each plot using the scipy.stats.linregress() function.

@author: kyurheekim
"""

import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit

# create subplots
fig = plt.figure()
plt1 = fig.add_subplot(221)
plt2 = fig.add_subplot(222)

# for the column "vaccinated"

# define function 
def V(t,V0,r1):
    vac = V0*np.exp(r1*t)
    return vac

# import data 
cols = np.loadtxt('/Users/kyurheekim/Desktop/Covid19Final.txt', skiprows=1)
tData = cols[:,0]
CData = cols[:,1]
VData = cols[:,2]

# define initial guess for model parameters
V0 = 67109
r1 = 0.0162
p1 = V0, r1

# fit data to exponential growth model
popt,pcov = curve_fit(V, tData, VData,p1)
p_stderr = np.sqrt(np.diag(pcov))

# plot the data
plt1.plot(tData, VData, linestyle='',marker='d',markersize=3.0, label='data')

# plot the theory
V0,r1 = popt
tTheory = np.linspace(0,100,10)
VTheory = V(tTheory,V0,r1)
plt1.plot(tTheory, VTheory, label = 'model')
plt1.set_xlabel('t [date]')
plt1.set_ylabel('Vaccinated')
plt1.set_title('Vaccinated Rate', fontsize = 10.0)
plt1.legend()


# for the column "confirmed"

# define function
def C(t,C0,r2):
    con = C0*np.exp(r2*t)
    return con

# define initial guess for model parameters
C0 = 1745361
r2 = -0.000807
p2 = C0, r2

# fit data to exponential growth model
popt,pcov = curve_fit(C, tData, CData,p2)
p_stderr = np.sqrt(np.diag(pcov))

# plot the data
plt2.plot(tData, CData, linestyle='', marker='d', markersize=3.0, label='data')

# plot the theory
C0,r2 = popt
tTheory2 = np.linspace(0,100,10)
CTheory = C(tTheory2,C0,r2)
plt2.plot(tTheory2, CTheory, label = 'model')
plt2.set_xlabel('t [date]')
plt2.set_ylabel('Confirmed Cases')
plt2.yaxis.set_label_position('right')
plt2.yaxis.tick_right()
plt2.set_title('Confirmed Cases Rate', fontsize = 10.0)
plt2.legend()

#showing the graphs
plt.show()
    
    
    
    
    
    
    
    
    
    
    
    


