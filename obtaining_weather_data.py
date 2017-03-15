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

# Put your weather underground apikey
wuapikey = '40d2bb7e63332cdf'

# Change to your own location.
coordinates = 'LATITUDE,LONGITUDE'

# Open the metadata
apiDic = pd.read_csv('~\\ML-energy-use\\SECRETS_DIRECTORY\\apiKeyDictionary.csv')


weather = pd.DataFrame()

# Obtain the necesary period
initdate = dt.datetime.utcfromtimestamp(apiDic.loc[(apiDic['lan_long']==coordinates),['start']].min())
enddate = dt.datetime.utcfromtimestamp(apiDic.loc[(apiDic['lan_long']==coordinates),['end']].max())
print(initdate.strftime('%Y%m%d'), enddate.strftime('%Y%m%d'))

# Call to weather underground
for day in pd.date_range(initdate,enddate, freq= 'd'):
  
    day = day.strftime('%Y%m%d')
    response = requests.get('http://api.wunderground.com/api/'+wuapikey+'/history_'+
                            str(day)+'/q/'+coordinates+'.json').content
                            

    decoded = json.loads(response)
    
        
    df = pd.DataFrame(decoded['history']['observations'])

# Uncoment to create a datetime from a list.
#        for j in range(df['date'].size):
#            df['date'][j] = pd.datetime(int(df['date'][j]['year']),int(df['date'][j]['mon']),
#                        int(df['date'][j]['mday']),int(df['date'][j]['hour']),int(df['date'][j]['min']))
    
    weather = weather.append(df)

print('Weather')
print(weather)

# Save data 
weather.to_csv('~\\ML-energy-use\\DATA_DIRECTORY\\WEATHER_DATA\\'+coordinates+'.csv')



