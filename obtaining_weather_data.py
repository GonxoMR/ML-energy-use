# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:47:10 2017

@author: Gonxo

This code imports the historical weather data from weather underground. 
You need to set up an account and the requests will consume one of the given raindrops, 
so collect as many as posible in one day. 
"""
import requests
import json
import pandas as pd
import datetime as dt
import os
from pandas.tseries.offsets import Day

wd = os.getcwd()

# Put your weather underground apikey
wuapikey = '40d2bb7e63332cdf'

# Change to your own location.
#coordinates = 'LATITUDE,LONGITUDE'

# Open the metadata
apiDic = pd.read_csv(os.path.join(wd,'SECRETS_DIRECTORY','apiKeyDictionary.csv'))

for coordinates in apiDic['lat_long']:
    
    weather = pd.DataFrame()

    enddate = dt.datetime.utcfromtimestamp(apiDic.loc[(apiDic['lat_long']==coordinates),['end']].max())
    
    if os.path.isfile(os.path.join(wd,'DATA_DIRECTORY','WEATHER_DATA','%s.csv' %coordinates)):
        weather = pd.DataFrame.from_csv(os.path.join(wd,'DATA_DIRECTORY','WEATHER_DATA','%s.csv' %coordinates))

        if 'date' in weather.columns:
            if dt.datetime.strptime(weather['date'].max(), '%Y-%m-%d %H:%M:%S').date() < enddate.date():
#                initdate = dt.datetime.utcfromtimestamp(apiDic.loc[(apiDic['lat_long']==coordinates),['start']].min())
                initdate = dt.datetime.strptime(weather['date'].max(), '%Y-%m-%d %H:%M:%S').date()+Day()
            
            else:
                print('There is already weather data for the coordinates %s.' %coordinates)
                continue
    
    else:
        initdate = dt.datetime.utcfromtimestamp(apiDic.loc[(apiDic['lat_long']==coordinates),['start']].min())
    
    print('Collecting weather data from %s to %s coordinates %s' %(initdate, enddate.date(), coordinates))
    
    # Obtain the necesary period
    # Call to weather underground
    for day in pd.date_range(initdate,enddate, freq= 'd'):
      
        day = day.strftime('%Y%m%d')
        response = requests.get('http://api.wunderground.com/api/'+wuapikey+'/history_'+
                                str(day)+'/q/'+coordinates+'.json')
                                
    
        decoded = json.loads(response.text)
        
            
        df = pd.DataFrame(decoded['history']['observations'])
    
    # Uncoment to create a datetime from a list.
        for j in range(df['date'].size):
            df['date'][j] = pd.datetime(int(df['date'][j]['year']),int(df['date'][j]['mon']),
                        int(df['date'][j]['mday']),int(df['date'][j]['hour']),int(df['date'][j]['min']))
    
        weather = weather.append(df)

    # Save data 
    weather.to_csv(os.path.join(wd,'DATA_DIRECTORY','WEATHER_DATA','%s.csv' %coordinates))



