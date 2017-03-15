# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:03:20 2017



@author: Gonxo
"""
import pandas as pd

#This function computes ANOVA tests over different seasonal periods. 
#The input is a feed of data and the desired period of seasonality. 
#The function computes the features quarterhourofday (0,95), dayofweek (0,6) and
#month(1,12).
#The output are the estatistic for the difference of means and its p-value to be 0.
def seasonalANOVA(HC, period='month'):
    import scipy.stats as stats
    
    if period == 'quarterhourofday':
        
        counter = 0
        array = []
        for i in pd.date_range('00:00', '23:45', freq='15min'):
            HC.loc[(HC.index.hour == i.hour) & (HC.index.minute == i.minute),'quarterhourofday'] = counter
            array.append( HC.loc[HC['quarterhourofday']== counter].values)
            counter += 1
        
        return (stats.f_oneway( HC.loc[HC['quarterhourofday']==0], HC.loc[HC['quarterhourofday']==1], HC.loc[HC['quarterhourofday']==2], HC.loc[HC['quarterhourofday']==3], HC.loc[HC['quarterhourofday']==4],
                                               HC.loc[HC['quarterhourofday']==5], HC.loc[HC['quarterhourofday']==6], HC.loc[HC['quarterhourofday']==7], HC.loc[HC['quarterhourofday']==8], HC.loc[HC['quarterhourofday']==9],
                                               HC.loc[HC['quarterhourofday']==10], HC.loc[HC['quarterhourofday']==11], HC.loc[HC['quarterhourofday']==12], HC.loc[HC['quarterhourofday']==13], HC.loc[HC['quarterhourofday']==14],
                                               HC.loc[HC['quarterhourofday']==15], HC.loc[HC['quarterhourofday']==16], HC.loc[HC['quarterhourofday']==17], HC.loc[HC['quarterhourofday']==18], HC.loc[HC['quarterhourofday']==19],
                                               HC.loc[HC['quarterhourofday']==20], HC.loc[HC['quarterhourofday']==21], HC.loc[HC['quarterhourofday']==22], HC.loc[HC['quarterhourofday']==23], HC.loc[HC['quarterhourofday']==24],
                                               HC.loc[HC['quarterhourofday']==25], HC.loc[HC['quarterhourofday']==26], HC.loc[HC['quarterhourofday']==27], HC.loc[HC['quarterhourofday']==28], HC.loc[HC['quarterhourofday']==29],
                                               HC.loc[HC['quarterhourofday']==30], HC.loc[HC['quarterhourofday']==31], HC.loc[HC['quarterhourofday']==32], HC.loc[HC['quarterhourofday']==33], HC.loc[HC['quarterhourofday']==34],
                                               HC.loc[HC['quarterhourofday']==35], HC.loc[HC['quarterhourofday']==36], HC.loc[HC['quarterhourofday']==37], HC.loc[HC['quarterhourofday']==38], HC.loc[HC['quarterhourofday']==39],
                                               HC.loc[HC['quarterhourofday']==40], HC.loc[HC['quarterhourofday']==41], HC.loc[HC['quarterhourofday']==42], HC.loc[HC['quarterhourofday']==43], HC.loc[HC['quarterhourofday']==44],
                                               HC.loc[HC['quarterhourofday']==45], HC.loc[HC['quarterhourofday']==46], HC.loc[HC['quarterhourofday']==47], HC.loc[HC['quarterhourofday']==48], HC.loc[HC['quarterhourofday']==49],
                                               HC.loc[HC['quarterhourofday']==50], HC.loc[HC['quarterhourofday']==51], HC.loc[HC['quarterhourofday']==52], HC.loc[HC['quarterhourofday']==53], HC.loc[HC['quarterhourofday']==54],
                                               HC.loc[HC['quarterhourofday']==55], HC.loc[HC['quarterhourofday']==56], HC.loc[HC['quarterhourofday']==57], HC.loc[HC['quarterhourofday']==58], HC.loc[HC['quarterhourofday']==59],
                                               HC.loc[HC['quarterhourofday']==60], HC.loc[HC['quarterhourofday']==61], HC.loc[HC['quarterhourofday']==62], HC.loc[HC['quarterhourofday']==63], HC.loc[HC['quarterhourofday']==64],
                                               HC.loc[HC['quarterhourofday']==65], HC.loc[HC['quarterhourofday']==66], HC.loc[HC['quarterhourofday']==67], HC.loc[HC['quarterhourofday']==68], HC.loc[HC['quarterhourofday']==69],
                                               HC.loc[HC['quarterhourofday']==70], HC.loc[HC['quarterhourofday']==71], HC.loc[HC['quarterhourofday']==72], HC.loc[HC['quarterhourofday']==73], HC.loc[HC['quarterhourofday']==74],
                                               HC.loc[HC['quarterhourofday']==75], HC.loc[HC['quarterhourofday']==76], HC.loc[HC['quarterhourofday']==77], HC.loc[HC['quarterhourofday']==78], HC.loc[HC['quarterhourofday']==79],
                                               HC.loc[HC['quarterhourofday']==80], HC.loc[HC['quarterhourofday']==81], HC.loc[HC['quarterhourofday']==82], HC.loc[HC['quarterhourofday']==83], HC.loc[HC['quarterhourofday']==84],
                                               HC.loc[HC['quarterhourofday']==85], HC.loc[HC['quarterhourofday']==86], HC.loc[HC['quarterhourofday']==87], HC.loc[HC['quarterhourofday']==88], HC.loc[HC['quarterhourofday']==79],
                                               HC.loc[HC['quarterhourofday']==90], HC.loc[HC['quarterhourofday']==91], HC.loc[HC['quarterhourofday']==92], HC.loc[HC['quarterhourofday']==93], HC.loc[HC['quarterhourofday']==74],
                                               HC.loc[HC['quarterhourofday']==95]))
    elif period == 'dayofweek':
        
        HC['dayofweek'] = HC.index.dayofweek
        
        return (stats.f_oneway(HC.loc[HC['dayofweek']== 0], HC.loc[HC['dayofweek']== 1], 
                                               HC.loc[HC['dayofweek']== 2], HC.loc[HC['dayofweek']== 3], 
                                               HC.loc[HC['dayofweek']== 4], HC.loc[HC['dayofweek']== 5],
                                               HC.loc[HC['dayofweek']== 6]))
    elif period == 'month':
        
        HC['month'] = HC.index.month
        
        return (stats.f_oneway(HC.loc[HC['month']== 1], HC.loc[HC['month']== 2],
                                               HC.loc[HC['month']== 8],
                                               HC.loc[HC['month']== 9], HC.loc[HC['month']== 10],
                                               HC.loc[HC['month']== 11], HC.loc[HC['month']== 12]))
    
    
    else:
        print('Error: the options for period are quarterhourofday, dayofweek or month')


# This function deseasonalizes a time series based on a period of quarter of hour, 
# day of the week and month of the year. 
# The input is the feed to deseasonalize and its naming id.
# The process of deseasonalization is describen in [...paper...]
# It computes for each periodicity:
#    1. The mean over the three aggregation periods.
#    2. The normalized series by dividing it with the previous mean.
#    3. The deseasonalized series dividing it by the median of the previos normalized values.
# UM_feedid is the final deseasonalized series. 
# It returns a data frame [len(feed),10] with the original feed and all the derivated series. 

def deseasonalize(feed, r_id):
    import matplotlib.pyplot as plt
    
    feed[r_id] = pd.DataFrame(feed)
    
    for i in range(3):
        if i == 0:
            aggPeriod = 'D'
            granularity = '15Min'
            y = feed[r_id]
#            winSize = 96
            
        elif i == 1:
            aggPeriod = granularity = 'W'
#            winSize = 672
            
                        
        elif i == 2:
            aggPeriod = granularity = 'M'
#            winSize = 2880
    
        avg = 'avg'+aggPeriod+'_'+r_id
        z = 'Z'+aggPeriod+'_'+r_id
        u = 'U'+granularity+'_'+r_id

# Daily average

        feed[avg] = feed[r_id].groupby(pd.TimeGrouper(aggPeriod)).transform('median')
# Moving average 1 day window
#        feed[avg] = feed[r_id].rolling(window=winSize,center=True).mean()

# Normalized series for 1 day
        feed[z] = feed[r_id] / feed[avg]
        
# Findind quarter of an hour means
        feed[u] = y/feed[z].groupby(pd.TimeGrouper(granularity)).transform('mean')
        
        y = feed[u]

    plt.plot(feed[r_id], '-', ms = 0.1)
    plt.plot(feed[u], '-', ms= 0.1)
    plt.plot(feed['avgD_'+r_id], '-', ms = 0.2)
    plt.plot(feed['avgW_'+r_id], '-', ms = 0.2)
    plt.plot(feed['avgM_'+r_id], '-', ms = 0.2)
    plt.show()
    
    return(feed)