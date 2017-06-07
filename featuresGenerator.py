# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 20:44:53 2017

@author: Gonxo
"""
import numpy as np
import mlfunctions as mlf
from sklearn import preprocessing
import os

def evalDate(x):
    """
    Function that evaluates and cleans the date from the weather data obtained in WU.
    """
    import pandas as pd
    x = eval(x)
    if x['min'] < '20':
        x['min'] = str(20)
    return (pd.datetime(int(x['year']), int(x['mon']), int( x['mday']), int( x['hour']), int( x['min'])-20))

def autoPCAFactor(X, threshold = 0.8):
    """
    This function automates the search of the number of factors to fit the PCA.

    Input:
        X: matrix to be decomposed.
        Threshold: minimum proportional variability to be explain by the components.

    Output:
        PCA fitted object with the number of factors that explain the minimum variance.
    """
    from sklearn.decomposition import PCA
    n_factors = 1
    pca = PCA(n_components=n_factors)
    pca.fit(X)
    while (pca.explained_variance_ratio_.sum() < 0.99):
        n_factors += 1
        pca = PCA(n_components=n_factors)
        pca.fit(X)
    return pca

def featureCreation(feed, window, h, grouper, dataDir, apiDic, r_id = None, longestfeed = False):
    import pandas as pd
    from bdateutil import isbday
    import holidays
    from sklearn.decomposition import PCA
    
    feed = pd.DataFrame(feed)
    r_lat_long = apiDic.loc[(apiDic['id']==int(r_id)), 'lat_long'][1]
    
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
    weather = pd.DataFrame.from_csv(os.path.join(dataDir,'WEATHER_DATA','%s.csv'  %r_lat_long.replace(" ", "")))
    
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

    weather = weather.loc[:,('conds', 'dewptm', 'fog', 'hail','hum', 
                      'precipm', 'pressurem','rain', 'snow', 'tempm', 'thunder',
                      'wdire', 'wgustm',  'windchillm', 'wspdm')]
    
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
     
    if (weather.index.min() < feed.index.min()):
        if (weather.index.max() < feed.index.max()):
            weather = weather.ix[feed.index.min():, :]
            feed = feed.ix[:weather.index.max(), :]
        else:
            weather = weather.ix[feed.index.min():feed.index.max(), :]
    else:
        if (weather.index.max() < feed.index.max()):
            feed = feed.ix[weather.index.min():weather.index.max(), :]
        else:
            feed = feed.ix[weather.index.min():, :]
            weather = weather.ix[:feed.index.max(), :]

    features, response = mlf.ts_to_mimo(feed.ix[:, 0], window, h)   

    n_factors = 1
    pca = PCA(n_components=n_factors)
    pca.fit(weather)
    while (pca.explained_variance_ratio_.sum() < 0.99):
        n_factors += 1
        pca = PCA(n_components=n_factors)
        pca.fit(weather)
        
    reduced = pd.DataFrame(pca.transform(weather))         

    c = np.zeros((features.shape[0], (h * len(reduced.columns))))
    
    for column in range(len(reduced.columns)):
        c[:, (column*h):((1+column)*h)] = mlf.weather_to_mimo(reduced.ix[:,column], window, h)

    features = np.concatenate((feed.ix[(window+h-1):, ('isworkingday',grouper,'hourofday','dayofweek','month')],c, features), axis=1)
    
    print('Features created')

    return (features, response)
