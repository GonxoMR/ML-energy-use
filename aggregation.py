# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 16:06:36 2017

@author: Gonxo

This file is merges the different grid and solar feed into a single feed. Also
it deseasonalizes the data in hour of the daya, day of the week, and month of 
the year. Finally it computes ANOVA tests to check seasonal variations significancy.

NOTE: This code has been writen in a windows machine. To use it in a linux
machine change the directory address form '\\' to '/'
"""
import pandas as pd
import time
import matplotlib.pyplot as plt


start = time.time()

# Set up the destination  and secrects directory
dataDir = 'C:\\Users\\Gonxo\\ML-energy-use\\DATA_DIRECTORY'
secretsDir = 'C:\\Users\\Gonxo\\ML-energy-use\\SECRETS_DIRECTORY'
apiDic = pd.read_csv(secretsDir+'\\apiKeyDictionary.csv')
store = pd.HDFStore('C:\\Users\\Gonxo\\ML-energy-use\\DATA_DIRECTORY\\aggregated_feeds.h5')

# Loading relevant data after removing NaNs adding grid power and solar power. 
# House consumption = Grid power + solar power
# HC = house_consumption
HC = pd.DataFrame()
with pd.HDFStore(dataDir+'\\15min_noNaNs_201703081045.h5') as hdf:
    keys = hdf.keys()
    
for index, row in apiDic.loc[:,['key','type','id']].iterrows(): #len(apiDic)):
#    print('/'+row['type']+str(row['id']))
    r_id = str(row['id'])
    if (('/'+row['type']+'_'+r_id) in keys) and (row['type'] != 'solar_power'):
        
        if row['type'] == 'house_consumption':

#           Original series
            HC[r_id] = pd.read_hdf(dataDir+'\\15min_noNaNs_201703081045.h5', str(row['type'])+'_'+r_id)
      
        elif (row['type']=='grid_power'):
            print(str(row['type'])+'_'+r_id)
            grid_feed = pd.read_hdf(dataDir+'\\15min_noNaNs_201703081045.h5', str(row['type'])+'_'+r_id)
            solar_id = apiDic.ix[(apiDic['key'] == row['key']) & (apiDic['type'] == 'solar_power')]['id'].values[0]
            solar_feed = pd.read_hdf(dataDir+'\\15min_noNaNs_201703081045.h5', 'solar_power_'+str(solar_id))
                  
            HC[r_id] = grid_feed+solar_feed
        
        
        HC[r_id] = HC[r_id].interpolate('from_derivatives') 
        
        result= deseasonalize(HC[r_id], r_id)


end = time.time()
print(end - start)