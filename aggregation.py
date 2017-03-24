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
import mlfunctions as mlf

start = time.time()

# Set up the destination  and secrects directory
dataDir = 'C:\\Users\\Gonxo\\ML-energy-use\\DATA_DIRECTORY'
secretsDir = 'C:\\Users\\Gonxo\\ML-energy-use\\SECRETS_DIRECTORY'
apiDic = pd.read_csv(secretsDir+'\\apiKeyDictionary.csv',sep=None, engine='python')
sourceFile = dataDir+'\\15min_noNaNs_201703081045.h5'

print('here')
store = pd.HDFStore('C:\\Users\\Gonxo\\ML-energy-use\\DATA_DIRECTORY\\aggregated_fs.h5')
print('here')
# Loading relevant data after removing NaNs adding grid power and solar power. 
# House consumption = Grid power + solar power
# HC = house_consumption

with pd.HDFStore(sourceFile) as hdf:
    keys = hdf.keys()
    
for index, row in apiDic.loc[:,['key','type','id']].iterrows(): 
    
    HC = pd.DataFrame()
    r_id = str(row['id'])
    
    if (('/'+row['type']+'_'+r_id) in keys) and (row['type'] != 'solar_power'):
        print(r_id)
        
        if row['type'] == 'house_consumption':

#           Original series
            HC = pd.read_hdf(sourceFile, str(row['type'])+'_'+r_id)

      
        elif (row['type']=='grid_power'):
            print(str(row['type'])+'_'+r_id)
            grid_feed = pd.read_hdf(sourceFile, str(row['type'])+'_'+r_id)
            grid_feed = grid_feed.interpolate('from_derivatives')
            solar_id = apiDic.ix[(apiDic['key'] == row['key']) & (apiDic['type'] == 'solar_power')]['id'].values[0]
            solar_feed = pd.read_hdf(sourceFile, 'solar_power_'+str(solar_id))
            solar_feed = solar_feed.interpolate('from_derivatives')
            HC = grid_feed+solar_feed
        
# This interpolate shouldn't be here is a patch to work with this data at the moment.
# When another batch of data will be collected and processed. This will be deleted as it is useless.
        HC = HC.interpolate('from_derivatives') 
        HC = mlf.tokwh(HC)
        result = mlf.deseasonalize(HC, r_id)
        
        print(HC.describe())
        
        store['feed_'+r_id] = result


end = time.time()
print(end - start)

r_id = str(20075)
feed = pd.read_hdf('C:\\Users\\Gonxo\\ML-energy-use\\DATA_DIRECTORY\\aggregated_fs.h5', '/feed_'+r_id)
