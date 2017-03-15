# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:55:12 2017

@author: Gonxo
"""
import random
import time
from pandas.tseries.offsets import *
import pandas as pd
import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt

def interpolationtest(hc, r_id):

#    print(hc.index.min())
    initstamp = hc.index.min()
    endstamp = hc.index.max()
    
    stampperiod = endstamp - initstamp
    
    randstamp = initstamp + (random.random() * stampperiod)
    
    randtime = dt(randstamp.year,randstamp.month, randstamp.day, randstamp.hour, 15*(randstamp.minute // 15))
    
#    print(randtime)
    
    while hc.loc[randtime:(randtime+ Day()),r_id].isnull().sum() != 0:
        randstamp = initstamp + random.random() * stampperiod
        randtime = pd.to_datetime(randstamp) 
    
    sliced = pd.DataFrame(hc[r_id])
    
    
    sliced[randtime : (randtime + Day())] = np.NaN
    
    linearinter = sliced.interpolate('linear')
    
    fromderivatesinterpolation = sliced.interpolate('from_derivatives')
    
    meaninterpolation = sliced.groupby(pd.TimeGrouper('W')).transform( lambda x: x.fillna(x.mean()))
    
    medianinterpolation = sliced.groupby(pd.TimeGrouper('2W')).transform( lambda x: x.fillna(x.median()))
    
    previousWeek = sliced
    
    previousWeek.ix[previousWeek.isnull().index] = sliced.ix[sliced.isnull().index-Week()].values

    errorSquared = pd.DataFrame()
    errorSquared['linear'] = (( linearinter.loc[randtime : (randtime + Day())].subtract( hc.loc[randtime : (randtime + Day()),r_id], axis=0)).sum().pow(2)/96) 
    
    errorSquared['derivatives'] = (( fromderivatesinterpolation.loc[randtime : (randtime + Day())].subtract( hc.loc[randtime : (randtime + Day()),r_id], axis=0)).sum().pow(2))/96 
    
    errorSquared['mean'] = (( meaninterpolation.loc[randtime : (randtime + Day())].subtract( hc.loc[randtime : (randtime + Day()),r_id], axis=0)).pow(2).sum())/96
    
    errorSquared['median'] = (( medianinterpolation.loc[randtime : (randtime + Day())].subtract( hc.loc[randtime : (randtime + Day()),r_id], axis=0)).sum().pow(2))/96
    
    errorSquared['previousWeek'] = (( previousWeek.loc[randtime : (randtime + Day())].subtract( hc.loc[randtime : (randtime + Day()),r_id], axis=0)).sum().pow(2))/96
   
#    print(linearinter == fromderivatesinterpolation)
#    
#    print(errorSquared_linear.values, errorSquared_derivatives.values, errorSquared_mean.values, errorSquared_median.values, errorSquared_previousWeek.values)
#    
#    plt.plot(sliced.loc[(randtime - 3*Day()) : (randtime + 4*Day())],'-',ms= 0.1)
#    plt.plot(hc.loc[randtime : (randtime + Day()),r_id],'-', ms = 0.1)
#    plt.plot(linearinter.loc[randtime : (randtime + Day())],'-', ms = 0.1)
#    plt.plot(fromderivatesinterpolation.loc[randtime : (randtime + Day())],'-', ms = 0.1)
#    plt.plot(meaninterpolation.loc[randtime : (randtime + Day())],'-', ms = 0.1)
#    plt.plot(medianinterpolation.loc[randtime : (randtime + Day())],'-', ms = 0.1)
#    plt.plot(previousWeek.loc[randtime : (randtime + Day())],'-', ms = 0.1)
#    plt.legend(['Series','Real_value','Linear','Derivates','mean', 'median', 'lastWeek'],mode='expand',borderaxespad=0.)
#    plt.show()
    return(errorSquared)

errors = pd.DataFrame()
for i in range(4000):
    errors = errors.append( interpolationtest(HC, r_id))


errors.to_csv('C:\\Users\\Gonxo\\errors.csv')
