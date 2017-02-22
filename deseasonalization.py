import numpy as np 
import pandas as pd
# This file is a data diagnosis and NA cleaning file. 
# For every data feed it performs the next actions:
#    1. For NaN periods that last <15min it coerces the group median on them.
#    2. It deletes the initial errors, i.e. numeric but eror values and NaNs.
#    3. Finding the groups of NaN left in the series. These are big enough to
#       not be corrected by the median proccess.
#    4. Computes descriptive statistics about the duration and frequency of
#       the NaN and operatve periods. 
#    
import time
from pandas.tseries.offsets import *
import matplotlib.pyplot as plt

secretsDir = 'SECRETS_DIRECTORY'
apiDic = pd.read_csv('~\\ML-energy-use\\'+secretsDir+'\\apiKeyDictionary.csv')
ids = apiDic['id']
type = apiDic['type']

# Need to know which feeds add toguether


start = time.time()
feeds = pd.DataFrame()

for i in range(2):#len(apiDic)):
    print(str(type[i])+'_'+str(ids[i]))
    
    # Obtaining the feed from the hdf5 file
    feeds = pd.read_hdf('C:\\Users\\Gonxo\\ML-energy-use\\DATA_DIRECTORY\\home_feeds.h5', str(type[i])+'_'+str(ids[i]))['unit']
     
    # Deleting NaNs at the beggining and end of te series.
    first = feeds.first_valid_index()
    last = feeds.last_valid_index()
    feeds = feeds.loc[first:last]
    
#	 1. Removing NA's by coercing the 15 min group median
    grouped = feeds.groupby(pd.TimeGrouper(freq='15min'))
    f = lambda x: x.fillna(x.median())
    no_Nas = grouped.transform(f)
    
#   2. Deleting initial setup problems: If measurement(t1) = 0 or
    # (abs(measurements(t1)) <= 25 and measurement(t2) = 0) 
    while (no_Nas.loc[no_Nas.first_valid_index()] == float(0) or
            (abs(no_Nas.loc[no_Nas.first_valid_index()]) <= float(25) and
            no_Nas.loc[no_Nas.first_valid_index()+(10 * Second())] == float(0))):
        # Droping 1st observation
        no_Nas = no_Nas.drop([no_Nas.first_valid_index()])
        first = no_Nas.first_valid_index()
    
    # Deleting NaNs at the beggining of the series after removing some erroneous observations
    first = no_Nas.first_valid_index()
    no_Nas = no_Nas.loc[first:last]
    
    
#   3. This retrieves the blocks of Nan in the even groups.
    block = (no_Nas.notnull().shift(1) !=  no_Nas.notnull()).astype(int).cumsum()
    
    no_Nas = pd.concat([no_Nas, block], axis=1)
    no_Nas.columns = ['unit','block']
    naGroups = no_Nas.reset_index().groupby(['block'])['index'].apply(np.array)

#   4. Computing statistics
    # Finding the number of groups of NaN in the series
    na_periods = np.floor(naGroups.size/2)
    
    # Finding the aumout of time the device is off for each period.
    na_time_periods = []
    for i in range(int(na_periods)):
        na_time_periods.append((naGroups[2*(i+1)].max()- naGroups[2*(i+1)].min()).astype('timedelta64[m]'))
    na_time_periods = pd.Series(na_time_periods)
    
    # Average time the device is off when it is off
    time_off_statistics = na_time_periods.describe()
    
    # Finding the aumout of time the device is off for each period.
    working_time_periods = []
    for i in range(int(na_periods)):
        working_time_periods.append((naGroups[2*(i+1)-1].max()- naGroups[2*(i+1)-1].min()).astype('timedelta64[m]'))
    working_time_periods = pd.Series(working_time_periods)
    
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
#    bins=[0,30,60,360,720,100000]
#    hist, bind_edges = np.histogram(na_time_periods.astype('timedelta64[m]'), bins=bins)
#    fig,ax = plt.subplots()
#    ax.bar(range(len(hist)),hist,width=1)
#    ax.set_xticks([0.5+i for i,j in enumerate(hist)])
#    ax.set_xticklabels(['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist)])
#    plt.show()

    
    grouped_data_noNas = no_Nas.groupby(pd.TimeGrouper(freq='15min')).mean()
    del grouped_data_noNas['block']
    
    grouped_data_noNas.plot()
    
    nulls = grouped_data_noNas.loc[grouped_data_noNas['unit'].isnull()]
    print(grouped_data_noNas.ix[nulls.index])
    print(grouped_data_noNas.ix[nulls.index - Week(),'unit'])
    grouped_data_noNas.ix[nulls.index] = grouped_data_noNas.ix[nulls.index-Week()].values#.apply(lambda x:  grouped_data_noNas.loc[x.index-Week(),'unit'])
    print('after')
    print(grouped_data_noNas.ix[nulls.index])
    
    grouped_data_noNas.plot()
    
#    #print( no_Nas.isnull().astype(int).groupby(no_Nas.notnull().astype(int).cumsum()).sum() )
#    print(first)
#    print( 'DataFrame: '+ str(feeds.count()))
#    print( 'NAs: '+ str(feeds.isnull().sum()))
    print( 'NoNAsSize: '+ str(no_Nas['unit'].size))
#    print(grouped_data_noNas['unit'])
#    print( 'NAs: '+ str(no_Nas.isnull().sum()))
#	
	# print( 'Groups mean:\n' +str(grouped.mean()) + ' - '+str(grouped_trans.mean()))

    
    # Getting last week values
#    no_Nas.loc[no_Nas['unit'].isnull(), 'unit'].apply(lambda x: no_Nas.loc[x.index - Week()])
    
    
    # print( no_Nas[selected.index.shift(1,freq='10s')])
#    print(no_Nas['unit'].isnull().sum())
    print(grouped_data_noNas['unit'].isnull().sum())

    
    
#    print(no_Nas.tail)
	
end = time.time()
print(end - start)