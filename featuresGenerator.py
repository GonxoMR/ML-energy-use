# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 20:44:53 2017

@author: Gonxo
"""
import pandas as pd
from bdateutil import isbday
import holidays 
from pandas.tseries.offsets import *
import numpy as np
import mlfunctions
from meteocalc import Temp, dew_point, heat_index
import datetime as dt

dataDir = 'C:\\Users\\Gonxo\\ML-energy-use\\DATA_DIRECTORY'
secretsDir = 'C:\\Users\\Gonxo\\ML-energy-use\\SECRETS_DIRECTORY'
apiDic = pd.read_csv(secretsDir+'\\apiKeyDictionary.csv', sep=None, engine='python')

def evalDate(x):
    x = eval(x)
    if x['min'] < '20':
        x['min'] = str(20)
    
#    print(x['year'], x['mon'], x['mday'], x['hour'], x['min'])
    return (pd.datetime(int(x['year']), int(x['mon']), int( x['mday']), int( x['hour']), int( x['min'])-20))

def featureCreation(feed, r_id):
    
# Quarter of hour
    counter = 0
    array = []
    for i in pd.date_range('00:00', '23:45', freq='15min'):
        feed.loc[(feed.index.hour == i.hour) & (feed.index.minute == i.minute),'quarterhourofday'] = counter
        array.append( feed.loc[feed['quarterhourofday']== counter].values)
        counter += 1

# Hour of day
    feed['hourofday'] = feed.index.hour
    
# Day of week
    feed['dayofweek'] = feed.index.dayofweek
    
# Month
    feed['month'] = feed.index.month

# Working day
#    f = np.vectorize(lambda x: isbday(x, holidays=holidays.UK(years=[2013,2014,2015,2016,2017])))
#    feed['isworkingday'] = f(feed.index.date)
#    
# Previous readings○
    for i in range(1,70):
        feed['t_%s' %i] = feed[r_id].shift(i, freq ='15min')

# Weather data
    for index, row in apiDic.loc[(apiDic['id']==int(r_id)), ['key','lat_long']].iterrows():
        print(index, row.lat_long)
        weather = pd.read_csv(dataDir+'\\WEATHER_DATA\\%s.csv' %row['lat_long'].replace(" ", ""))
        
        # Converting text date into datetime       
        weather['cleandate'] = weather['utcdate'].apply(lambda x: evalDate(x))
        
        weather.index = weather['cleandate']
        # Deleting irrelevant columns   
        if 'date' in weather.columns:
            del weather['date']
            
        if 'date.1' in weather.columns:
            del weather['date.1']
        
        if 'utcdate' in weather.columns:
            del weather['utcdate']
        
        if 'Unnamed: 0' in weather.columns:
            del weather['Unnamed: 0']
        
        # Droping duplicates
        weather = weather.drop_duplicates(subset='cleandate')
        
        weather = weather.reindex(pd.date_range(weather['cleandate'].min(), weather['cleandate'].max(), freq='15min')
                                  , method='backfill')
        
        weather['heatindex'] = weather.apply(lambda x: heat_index(Temp(x['tempi'],'f'),x['hum']), axis=1).astype('float')
        
        weather = weather.loc[:,('conds', 'dewptm', 'fog', 'hail', 'heatindexm','hum', 
                          'precipm', 'pressurem','rain', 'snow', 'tempm', 'thunder',
                          'wdird', 'wdire', 'wgustm',  'windchillm', 'wspdm', 'heatindex')]
        
        weather.replace([-9999.0,-999.0],[np.nan,np.nan], inplace=True)
        
    
    feed = pd.merge(feed, weather, left_index=True, right_index=True)
    
    feed.loc[:,feed.dtypes=='object'] = feed.loc[:,feed.dtypes=='object'].apply(lambda x: x.astype('category'))
    
    return (feed)

feed = featureCreation(feed, r_id)



