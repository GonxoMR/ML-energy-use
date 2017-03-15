# This file is a data diagnosis and NA cleaning file. 
# For every data feed it performs the next actions:
#    1. For NaN periods that last <15min it coerces the group median on them.
#    2. It deletes the initial errors, i.e. numeric but eror values and NaNs.
#    3. Finding the groups of NaN left in the series. These are big enough to
#       not be corrected by the median proccess.
#    4. Computes descriptive statistics about the duration and frequency of
#       the NaN and operatve periods. 

import numpy as np 
import pandas as pd 
import time
from pandas.tseries.offsets import *
import matplotlib.pyplot as plt

# This function deletes the initial setup problems: If measurement(t1) = 0 or
# (abs(measurements(t1)) <= 25 and measurement(t2) = 0) 
def cleanBeggining(no_Nas):
    while (no_Nas.loc[no_Nas.first_valid_index()] == float(0) or
            (no_Nas.loc[no_Nas.first_valid_index()] <= float(15) and
            no_Nas.loc[no_Nas.first_valid_index()+(10 * Second())] <= float(0))):
        # Droping 1st observation
        no_Nas = no_Nas.drop([no_Nas.first_valid_index()])
        no_Nas = no_Nas.loc[no_Nas.first_valid_index():last]
    
    return (no_Nas.first_valid_index())
    
# Set up the destination  and secrects directory
dataDir = 'DATA_DIRECTORY'
secretsDir = 'SECRETS_DIRECTORY'
apiDic = pd.read_csv('~\\ML-energy-use\\'+secretsDir+'\\apiKeyDictionary.csv')
ids = apiDic['id']
type = apiDic['type']


# Counting time of proccessing
start = time.time()

# Creating a saving file for after processing
store = pd.HDFStore('C:\\Users\\Gonxo\\ML-energy-use\\DATA_DIRECTORY\\15min_noNaNs_201703081045.h5')

# Empty data frame to store previous feeds
feeds = pd.DataFrame()

# Looping for all the different feeds individually
for i in range(len(apiDic)):
    print(str(type[i])+'_'+str(ids[i]))
    
    # Obtaining the feed from the hdf5 file
    feeds = pd.read_hdf('C:\\Users\\Gonxo\\ML-energy-use\\DATA_DIRECTORY\\home_feeds.h5', str(type[i])+'_'+str(ids[i]))['watts_hour']
     
    # Deleting NaNs at the beggining and end of te series.
    first = feeds.first_valid_index()
    last = feeds.last_valid_index()
    feeds = feeds.loc[first:last]
    
#	 1. Removing NA's by coercing the 15 min group mean
    grouped = feeds.groupby(pd.TimeGrouper(freq='15min'))
    no_Nas = grouped.transform(lambda x: x.fillna(x.mean()))
    
#   2.Deleting NaNs at the beggining of the series after removing some erroneous observations
    first = cleanBeggining(no_Nas)
    no_Nas = no_Nas.loc[first:last]
    
    
#   3. This retrieves the blocks of Nan in the even groups.
    block = (no_Nas.notnull().shift(1) !=  no_Nas.notnull()).astype(int).cumsum()
    no_Nas = pd.concat([no_Nas, block], axis=1)
    no_Nas.columns = ['watts_hour','block']
    naGroups = no_Nas.reset_index().groupby(['block'])['index'].apply(np.array)

#   4. Computing statistics
    # Finding the number of groups of NaN in the series
    na_periods = np.floor(naGroups.size/2)
    
    # Finding the aumout of time the device is off for each period.
    na_time_periods = []
    for j in range(int(na_periods)):
        na_time_periods.append((naGroups[2*(j+1)].max()- naGroups[2*(j+1)].min()).astype('timedelta64[m]'))
    na_time_periods = pd.Series(na_time_periods, index = range(1, int(na_periods)*2, 2))

    # Average time the device is off when it is off
    time_off_statistics = na_time_periods.describe()
    
    # Finding the aumout of time the device is working for each period.
    working_time_periods = []
    for j in range(int(na_periods)):
        working_time_periods.append((naGroups[2*(j+1)-1].max()- naGroups[2*(j+1)-1].min()).astype('timedelta64[m]'))
    working_time_periods = pd.Series(working_time_periods)
    
    if len(working_time_periods):
        # Average time the device is off when it is off
        time_working_statistics = working_time_periods.describe()
        
        # NOTE: 
        # Also proportion of time being off from total of time.
        proportion_Nas = (no_Nas.isnull().sum() / no_Nas.size)*100
            
        print('Number of NaN periods: '+str(na_periods)+
              '\nTime unoperative:\n' +str(time_off_statistics)+
              '\nTime operative:\n'+ str(time_working_statistics)+
              '\nProportion data that is NA: '+ str(proportion_Nas[0])+'%')
        
        # Histogram of length of periods of time unoperative. <30min, 30m:1h , 1h:6h, 6h:12, >12
        bins=[0,30,60,360,720,100000]
        hist, bind_edges = np.histogram(na_time_periods.astype('timedelta64[m]'), bins=bins)
        fig,ax = plt.subplots()
        ax.bar(range(len(hist)),hist,width=1)
        ax.set_xticks([0.5+i for i,j in enumerate(hist)])
        ax.set_xticklabels(['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist)])
        plt.show()

# 5. Deleting part of series that have 
        no_Nas = no_Nas['watts_hour']
        long_NaNs = na_time_periods.loc[na_time_periods.astype('timedelta64[D]') > 14].index
        if len(long_NaNs):
            for j in long_NaNs:
                no_Nas = no_Nas.loc[naGroups[long_NaNs+4].values[0].max().astype('datetime64[ns]'):last]
        
            no_Nas = no_Nas.loc[cleanBeggining(no_Nas):last]
            
        grouped_data_noNas = no_Nas.groupby(pd.TimeGrouper(freq='15min')).sum()
    
    #   Negative values in house_consumption or grid_power are considered as errors
    #   and deleted.
        if (str(type[i]) == 'grid_power' or str(type[i]) == 'house_consumption'):
            grouped_data_noNas.loc[grouped_data_noNas < 0] = np.nan
        
    #    Ploting beffore cleaning Nans
        plt.figure(1)
        plt.subplot(211)
        grouped_data_noNas.plot()
# 5. Grouping data and imputing interpolation method.
    #   Computing NaN as the previous week value
        grouped_data_noNas.ix[nulls.index] = grouped_data_noNas.ix[nulls.index-Week()].values
        
        plt.subplot(212)
        grouped_data_noNas.plot()
        plt.show()
        
    #    # print( no_Nas[selected.index.shift(1,freq='10s')])
    ##    print(no_Nas['watts_hour'].isnull().sum())
        print(grouped_data_noNas.isnull().sum())
        # Saving individual data frames into the hf5 file. Each feed is individually saved with its type.
        store[str(type[i])+'_'+str(ids[i])] = grouped_data_noNas
    else:
        print ('This feed is empty')
        
store.close()	
end = time.time()
print(end - start)
