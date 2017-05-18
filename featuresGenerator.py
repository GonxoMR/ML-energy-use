# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 20:44:53 2017

@author: Gonxo
"""

from pandas.tseries.offsets import *
import numpy as np
import mlfunctions as mlf
from meteocalc import Temp, dew_point, heat_index
import datetime as dt
from sklearn import preprocessing
import os

def evalDate(x):
    import pandas as pd
    x = eval(x)
    if x['min'] < '20':
        x['min'] = str(20)
    
#    print(x['year'], x['mon'], x['mday'], x['hour'], x['min'])
    return (pd.datetime(int(x['year']), int(x['mon']), int( x['mday']), int( x['hour']), int( x['min'])-20))


def featureCreation(feed, window, h, grouper, dataDir, apiDic, r_id = None, longestfeed = False):
    import pandas as pd
    from bdateutil import isbday
    import holidays 
    
    feed = pd.DataFrame(feed)
    # print(feed)
    
# Quarter of hour
    counter = 0
    array = []
    for i in pd.date_range('00:00', '23:45', freq=grouper):
        feed.loc[(feed.index.hour == i.hour) & (feed.index.minute == i.minute),grouper] = counter
        array.append( feed.loc[feed[grouper]== counter].values)
        counter += 1

# Hour of day
    feed['hourofday'] = feed.index.hour
    
# Day of week
    feed['dayofweek'] = feed.index.dayofweek
    
# Month
    feed['month'] = feed.index.month

# Working day
    f = np.vectorize(lambda x: isbday(x, holidays=holidays.UK(years=[2013,2014,2015,2016,2017])))
    feed['isworkingday'] = f(feed.index.date)
  
# Weather data
    features, response = mlf.ts_to_mimo(feed.ix[feed.first_valid_index():, 0], window, h)            

    if longestfeed == False:
#        print(feed.ix[feed.first_valid_index():, 0])
        
        for index, row in apiDic.loc[(apiDic['id']==int(r_id)), ['key','lat_long']].iterrows():
            
            weather = pd.DataFrame.from_csv(os.path.join(dataDir,'WEATHER_DATA','%s.csv'  %row['lat_long'].replace(" ", "")))
            
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
            
            weather = weather.reindex(pd.date_range(weather['cleandate'].min(), weather['cleandate'].max(), freq=grouper))#, method='backfill')
            
            weather['heatindex'] = weather.apply(lambda x: heat_index(Temp(x['tempi'],'f'),x['hum']), axis=1).astype('float')
            
            weather = weather.loc[:,('conds', 'dewptm', 'fog', 'hail','hum', 
                              'precipm', 'pressurem','rain', 'snow', 'tempm', 'thunder',
                              'wdird', 'wdire', 'wgustm',  'windchillm', 'wspdm', 'heatindex')]
            
            weather.loc[:,'conds'] = weather.loc[:,'conds'].fillna('Unknown')
            weather.loc[:,'wdire'] = weather.loc[:,'wdire'].fillna('Variable')
            
            le = le2 = preprocessing.LabelEncoder()
            le.fit(weather['conds'])
            weather['conds'] = le.transform(weather['conds'])
            le2.fit(weather['wdire'])
            weather['wdire'] = le2.transform(weather['wdire'])
            
            weather.replace([-9999.0,-999.0],[np.nan,np.nan], inplace=True)
            
            weather.loc[:, ('precipm','wgustm')] = weather.loc[:, ('precipm','wgustm')].fillna(0)
            
            weather.windchillm = weather.windchillm.fillna(weather.tempm)
            
            weather = weather.interpolate()
            
            if (weather.shape[0] < feed.shape[0]):
                if (feed.shape[0] - weather.shape[0] < (window + h)):
                    features = features[max(feed.index.get_loc(weather.index.min()-(window + h - 1)),0):feed.index.get_loc(weather.index.max()), :]
                    response = response[max(feed.index.get_loc(weather.index.min()-(window + h - 1)),0):feed.index.get_loc(weather.index.max()), :]
                    feed = feed.ix[weather.index.min():weather.index.max(),:]
                else:
                    features = features[feed.index.get_loc(weather.index.min()):(feed.index.get_loc(weather.index.max())+1), :]
                    response = response[feed.index.get_loc(weather.index.min()):(feed.index.get_loc(weather.index.max())+1), :]
                    feed = feed.ix[weather.index.min():weather.index.max(),:]
                    features = np.concatenate((feed.ix[:, ('isworkingday',grouper,'hourofday','dayofweek','month')],weather, features), axis=1)
            else:
                weather = weather.loc[feed.index[(window + h-1)]:feed.index.max()]

                weather = weather.values
                features = np.concatenate((feed.ix[(window + h -1):, ('isworkingday',grouper,'hourofday','dayofweek','month')],weather, features), axis=1)
            print('Features created')
    else:
        
        features, response = mlf.ts_to_mimo(feed.ix[feed.first_valid_index():, 0], window, h)
    
        row = apiDic.loc[(apiDic['id']==longestfeed), ('key','lat_long')]
        weather = pd.read_csv(dataDir+'\\WEATHER_DATA\\%s.csv' %str(row['lat_long'].values).replace(" ", "").replace("['", "").replace("']", "") )
            
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
        
        weather = weather.loc[:,('conds', 'dewptm', 'fog', 'hail','hum', 
                          'precipm', 'pressurem','rain', 'snow', 'tempm', 'thunder',
                          'wdird', 'wdire', 'wgustm',  'windchillm', 'wspdm', 'heatindex')]
        
        weather.loc[:,'conds'] = weather.loc[:,'conds'].fillna('Unknown')
        weather.loc[:,'wdire'] = weather.loc[:,'wdire'].fillna('Variable')
        
        le = le2 = preprocessing.LabelEncoder()
        le.fit(weather['conds'])
        weather['conds'] = le.transform(weather['conds'])
        le2.fit(weather['wdire'])
        weather['wdire'] = le2.transform(weather['wdire'])
        
        weather.replace([-9999.0,-999.0],[np.nan,np.nan], inplace=True)
        
        weather.loc[:, ('precipm','wgustm')] = weather.loc[:, ('precipm','wgustm')].fillna(0)
        
        weather.windchillm = weather.windchillm.fillna(weather.tempm)
        
        weather = weather.interpolate()
        
        weather = weather[feed.index[(window + h-1)]:feed.index.max()]
        
        weather = weather.values
        
        features = np.concatenate((feed.ix[(window + h -1):, ('activefeeds','isworkingday','quarterhourofday','hourofday','dayofweek','month')],weather, features), axis=1)

#    print(feed.ix[(window + h ):, ('isworkingday','quarterhourofday','hourofday','dayofweek','month')].shape, weather.shape, features.shape)
    
    return (features, response)




