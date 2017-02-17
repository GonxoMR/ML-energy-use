import numpy as np 
import pandas as pd
import time
from pandas.tseries.offsets import *

secretsDir = 'SECRETS_DIRECTORY'
apiDic = pd.read_csv('~\\ML-energy-use\\'+secretsDir+'\\apiKeyDictionary.csv')
ids = apiDic['id']
type = apiDic['type']

# Need to know which feeds add toguether
# Grouping data into 15 minutes periods


start = time.time()
feeds = pd.DataFrame()
i = 0
for i in range(2):#len(apiDic)):
    print(str(type[i])+'_'+str(ids[i]))
    
    feeds = pd.read_hdf('C:\\Users\\Gonxo\\ML-energy-use\\DATA_DIRECTORY\\home_feeds.h5', str(type[i])+'_'+str(ids[i]))['unit']
    
    
    first = feeds.first_valid_index()
    last = feeds.last_valid_index()
    feeds = feeds.loc[first:last]
    
    
	# Removing NA's by coercing the group median
    grouped = feeds.groupby(pd.TimeGrouper(freq='15min'))
	
    f = lambda x: x.fillna(x.median())
	
    no_Nas = grouped.transform(f)
    
    # Deleting initial setup problems: If measurement(t1) = 0 or
    # (abs(measurements(t1)) <= 25 and measurement(t2) = 0) 
    while (no_Nas.loc[no_Nas.first_valid_index()] == float(0) or
            (abs(no_Nas.loc[no_Nas.first_valid_index()]) <= float(25) and
            no_Nas.loc[no_Nas.first_valid_index()+(10 * Second())] == float(0))):
        # Droping 1st observation
        no_Nas = no_Nas.drop([no_Nas.first_valid_index()])
        first = no_Nas.first_valid_index()
    
    # Deleting all NaNs at the beggining of the series
    first = no_Nas.first_valid_index()
    no_Nas = no_Nas.loc[first:last]
    
    print(first)
    
    # This retrieves the blocks of Nan in the even groups.
    block = (no_Nas.notnull().shift(1) !=  no_Nas.notnull()).astype(int).cumsum()
    
    no_Nas = pd.concat([no_Nas, block], axis=1)
    no_Nas.columns = ['unit','block']
    naGroups = no_Nas.reset_index().groupby(['block'])['index'].apply(np.array)
    
    # This is the initial point of the block of NaNs
    # naGroups[even].min()
    # This is the initial point of the block of NaNs
    # naGroups[even].max()

    # Computing statistics
    # Finding the number of groups of NaN in the series
    na_periods = np.floor(naGroups.size/2)
    
    # Finding the aumout of time the device is off for each period.
    na_time_periods = []
    for i in range(int(na_periods)):
        na_time_periods.append((naGroups[2*(i+1)].max()- naGroups[2*(i+1)].min()).astype('timedelta64[m]'))
    
    # Average time the device is off when it is off
    avg_time_off = np.mean(na_time_periods)
    
    # NOTE: Would be interesting to see the span between periods of NA
    # i.e. span of time between the beggining and end of odd periods
    # Also proportion of time being off from total of time. 
    # Lenght of series / times off
    # Histogram of length of periods of time. <30min, 30m:1h , 1h:6h, 6h:12, 12:24, >24
    
    print('Number of NaN periods: '+str(na_periods)+
          '\nAverage time unoperative: ' +str(np.mean(na_time_periods)))
#    grouped_data_noNas = no_Nas.groupby(pd.TimeGrouper(freq='15min'))
#    
#    selected = no_Nas.ix[np.random.choice(no_Nas.loc[~no_Nas.isnull()].index, 20)]
#    
#    #print( no_Nas.isnull().astype(int).groupby(no_Nas.notnull().astype(int).cumsum()).sum() )
#    print(first)
#    print( 'DataFrame: '+ str(feeds.count()))
#    print( 'NAs: '+ str(feeds.isnull().sum()))
#    print( 'NoNAsSize: '+ str(no_Nas.size))
#    print( 'NAs: '+ str(no_Nas.isnull().sum()))
#	
	# print( 'Groups mean:\n' +str(grouped.mean()) + ' - '+str(grouped_trans.mean()))

	#nulls = no_Nas.loc[no_Nas.isnull()]
    
    # Getting last week values
    #last_Week = no_Nas[selected.index - Week()]
    
    # print( no_Nas[selected.index.shift(1,freq='10s')])
    # print(last_Week)
    #print(grouped_data_noNas.tail())
#    print(no_Nas.tail)
	
end = time.time()
print(end - start)