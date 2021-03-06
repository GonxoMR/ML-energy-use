# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:03:20 2017



@author: Gonxo
"""
import pandas as pd
import numpy as np 

def getting_saving_data(dataDir, secretsDir, apiDic, dataFile):
    """
    This function gets the data for the given apikeys and feeds. It stores the data
    and deletes the apikeys that are empty. Its output is saved in files, 
    making a local copy of the feeds and updating the apikey dictionary. 
    """
    import requests
    import numpy as np
    import time
    types = np.unique(apiDic['type'])
    # Creating a saving file
    store = pd.HDFStore('%s\\%s.h5' %(dataDir, dataFile))
    
    # Looping for all types of feed
    for typ in types:
    
        for index, row in apiDic.loc[(apiDic['type']==typ),['key','id']].iterrows():
    		
    		# NOTE: adding troubleshooting might be interesting. 
    		# Obtaining streamed data
            response = requests.get("https://emoncms.org/feed/export.json?apikey=%s&id=%s&start=0&CCBYPASS=H8F47SBDEJ" %(row['key'],row['id']))
    		
    		# Obtaining meta data for error detection and time generation
            meta = eval(requests.get("https://emoncms.org/feed/timevalue.json?apikey=%s&id=%s" %(row['key'],row['id'])).content)
    
    		# Checking data consistancy in bytes
            if not len(response.content):
                print('Error: the feed is empty. Deleting it. Feed: %s' %row['id']) 
                del apiDic[index]
                
            elif len(response.content) % 4 == 0 or (len(response.content)-2) % 4 == 0:
                    
    			# Creating a numpy array from data steam. Decodification is done here.
                if (len(response.content)-2) % 4 == 0:
                    array =  np.fromstring(response.content[2:], 'float32')
                else:
                    array = np.fromstring(response.content, 'float32')
    			
    			# Finding starting time point. 
    			# It is computed as the time value of the last observation minus the size of the data
                end_point = int(meta['time'])
                start_point = end_point - array.size*10
    			
    			# Checking if the obtained data stream has the same value as the last observation in the server. 
                if array[-1] != float(meta['value']):
                    print("Error: The last value of the array do not match the last value obtained from the feed: "+str(row['id']))
                    print('Last value array: %s\nLast value feed: %s' %(array[-1],meta['value']))
                    print(time.time())
    			# Giving time to the data observations
                df = pd.DataFrame({"watts":array},index= pd.date_range(pd.to_datetime(start_point+10, unit='s'),pd.to_datetime(end_point, unit='s'), freq='10S'))
                apiDic.loc[index,'start'] = start_point
                apiDic.loc[index, 'end'] = end_point	
                
            # Saving individual data frames into the hf5 file. Each feed is individually saved with its type.
                store['%s_%s' %(typ,row['id'])] = df
    							
            else:
                print("Error: The number of elements recieved is: %i which is not multiple of 4. Decodification is not posible. Feed: %s" %(len(response.content), row['id']))

    apiDic.to_csv('%s\\apiKeyDictionary.csv' %secretsDir)

def cleanBeggining(feed):
    """
    This function deletes the initial setup problems: If measurement(t1) = 0 or
    (abs(measurements(t1)) <= 25 and measurement(t2) = 0)
            
    """
    from pandas.tseries.offsets import Second
    
    feed = pd.DataFrame(feed)

    while (((feed.ix[feed.first_valid_index()] == float(0))[0])  or (((feed.ix[feed.first_valid_index()] <= float(15))[0]) and ((feed.ix[feed.first_valid_index()+(10 * Second())] == float(0))[0]))):
        # Droping 1st observation
        feed = feed.drop([feed.first_valid_index()])
        feed = feed.loc[feed.first_valid_index():feed.last_valid_index()]
    
    return (feed.first_valid_index())



def cleaning_nans(feed, r_type):
    """
    This function gives back a feed clean of Nas to be use in the later procedures.
    
    """
    feed = pd.DataFrame(feed)
    first = feed.first_valid_index()
    last = feed.last_valid_index()
    feed = feed.loc[first:last]
    
#   Negative values in house_consumption or grid_power are considered as errors and deleted.
    if(r_type == 'house_consumption'):
        feed.ix[feed.ix[:,0] < 0, 'watts'] = np.nan
    if(r_type == 'solar_power'):
        feed.ix[feed.ix[:,0] < 0, 'watts'] = 0.00001

#	 1. Removing NA's by coercing the 15 min group mean
    no_Nas = feed.groupby(pd.TimeGrouper(freq='15min')).transform(lambda x: x.fillna(x.mean()))

#   2.Deleting NaNs at the beggining of the series after removing some erroneous observations
    first = cleanBeggining(no_Nas)
    no_Nas = no_Nas.loc[first:last]

#   3. This retrieves the blocks of Nan in the even groups.
#    block = (no_Nas.notnull().shift(1) !=  no_Nas.notnull()).astype(int).cumsum()
#    no_Nas = pd.concat([no_Nas, block], axis=1)
#    no_Nas.columns = ['unit','block']
#    print(no_Nas)
#    naGroups = no_Nas.reset_index().groupby(['block'])['index'].apply(np.array)
#    
#    print(naGroups)

##   4. Computing statistics
#    # Finding the number of groups of NaN in the series
#    na_periods = np.floor(naGroups.size/2)
#    
#    # Finding the aumout of time the device is off for each period.
#    na_time_periods = []
#    for j in range(int(na_periods)):
#        na_time_periods.append((naGroups[2*(j+1)].max()- naGroups[2*(j+1)].min()).astype('timedelta64[m]'))
#    na_time_periods = pd.Series(na_time_periods, index = range(1, int(na_periods)*2, 2))
#    
#    # Finding the aumout of time the device is working for each period.
#    working_time_periods = []
#    for j in range(int(na_periods)):
#        working_time_periods.append((naGroups[2*(j+1)-1].max()- naGroups[2*(j+1)-1].min()).astype('timedelta64[m]'))
#    working_time_periods = pd.Series(working_time_periods)
          

# 3. Deleting part of series that have long periods of NA
    if len(no_Nas.notnull()):
#        no_Nas = no_Nas['watts']
#        no_Nas = no_Nas['unit']
#        long_NaNs = na_time_periods.loc[na_time_periods.astype('timedelta64[D]') > 14].index
#        if len(long_NaNs):
#            for j in long_NaNs:
#                no_Nas = no_Nas.loc[naGroups[long_NaNs+4].values[0].max().astype('datetime64[ns]'):last]
#        
#            no_Nas = no_Nas.loc[cleanBeggining(no_Nas):last]

        no_Nas = no_Nas.groupby(pd.TimeGrouper(freq='2W')).transform( lambda x: x.fillna(x.mean()))
        no_Nas = no_Nas.interpolate('quadratic')
    
        return no_Nas
            
    else:
        print ('This feed is empty')
        
def aggregation(r_key,r_type,r_id, sourceFile, apiDic):
    """
     Loading relevant data and after removing NaNs it adds grid power to solar power. 
     House consumption = Grid power + solar power 
    """
    
    with pd.HDFStore(sourceFile) as hdf:
        keys = hdf.keys()
               
    if (('/%s_%s' %(r_type,r_id) in keys)):
        
#        if (r_type == 'house_consumption') or (r_type == 'solar_power'):

#           Original series
        result = pd.read_hdf(sourceFile, '%s_%s' %(r_type,r_id))
        result = result.ix[result.index > pd.datetime(2014,1,1),:]
        
        result = cleaning_nans(result, r_type)
#            result = result.groupby(pd.TimeGrouper('W')).transform( lambda x: x.fillna(x.mean()))
            
#      
#        elif (r_type=='grid_power'):
##                print(str(r_type)+'_'+r_id)
#            grid_feed = pd.read_hdf(sourceFile, '%s_%s' %(r_type,r_id))
#            grid_feed = cleaning_nans(grid_feed, r_type)
##            grid_feed = grid_feed.groupby(pd.TimeGrouper('W')).transform( lambda x: x.fillna(x.mean()))
#            solar_id = apiDic.ix[(apiDic['key'] == r_key) & (apiDic['type'] == 'solar_power')]['id'].values[0]
#            solar_feed = pd.read_hdf(sourceFile, 'solar_power_%s' %str(solar_id))
#            solar_feed = cleaning_nans(solar_feed, r_type)
##            solar_feed = solar_feed.groupby(pd.TimeGrouper('W')).transform( lambda x: x.fillna(x.mean()))
#            result = grid_feed+solar_feed
#        
        return result


def seasonalANOVA(HC, period='month'):
    """
    This function computes ANOVA tests over different seasonal periods. 
    The input is a feed of data and the desired period of seasonality. 
    The function computes the features quarterhourofday (0,95), dayofweek (0,6) and
    month(1,12).
    The output are the estatistic for the difference of means and its p-value to be 0.
    """
    
    import scipy.stats as stats
    HC = pd.DataFrame(HC, index = HC.index)
    
    
    if period == 'quarterhourofday':
        
        counter = 0
        array = []
        for i in pd.date_range('00:00', '23:45', freq='15min'):
            HC.loc[(HC.index.hour == i.hour) & (HC.index.minute == i.minute),'quarterhourofday'] = counter
            array.append( HC.loc[HC['quarterhourofday']== counter].values)
            counter += 1
        
        watts = HC.columns[0]
        return (stats.f_oneway( *[HC.loc[HC['quarterhourofday']==i, watts] for i in range(96)]))
    
    if period == 'halfhourofday':
        
        counter = 0
        array = []
        for i in pd.date_range('00:00', '23:45', freq='30min'):
            HC.loc[(HC.index.hour == i.hour) & (HC.index.minute == i.minute),'halfhourofday'] = counter
            array.append( HC.loc[HC['halfhourofday']== counter].values)
            counter += 1
        
        watts = HC.columns[0]
        return (stats.f_oneway( *[HC.loc[HC['halfhourofday']==i, watts] for i in range(49)]))

    elif period == 'dayofweek':
        
        HC['dayofweek'] = HC.index.dayofweek
        watts = HC.columns[0]
        return (stats.f_oneway( *[HC.loc[HC['dayofweek']==i, watts] for i in range(7)]))

    elif period == 'month':
        
        HC['month'] = HC.index.month
        watts = HC.columns[0]
        return (stats.f_oneway( *[HC.loc[HC['month']==i, watts] for i in range(1,13)]))

    elif period == 'weekofyear':
        HC['weekofyear'] = HC.index.weekofyear
        watts = HC.columns[0]
        return (stats.f_oneway( *[HC.loc[HC['weekofyear']==i, watts] for i in range(1,54)]))
 
    elif period == 'semester':
        HC['semester'] = [get_semester(x) for x in HC.index]
        watts = HC.columns[0]
        return (stats.f_oneway(HC.loc[HC['semester']== 0, watts], HC.loc[HC['semester']== 1]))
    
    elif period == 'dayofyear':
        HC['dayofyear'] = HC.index.dayofyear
        watts = HC.columns[0]
        return (stats.f_oneway( *[HC.loc[HC['dayofyear']==i, watts] for i in range(367)]))

    elif period == 'quarter':
        HC['quarter'] = HC.index.quarter
        watts = HC.columns[0]
        return (stats.f_oneway( *[HC.loc[HC['quarter']==i, watts] for i in range(1,4)]))
    else:
        print('Error: the options for period are quarterhourofday, dayofweek or month')


def quarterofhour(feed,grouper):
    feed = pd.DataFrame(feed)
    counter = 0
    array = []
    for i in pd.date_range('00:00', '23:45', freq=grouper):
        feed.loc[(feed.index.hour == i.hour) & (feed.index.minute == i.minute),'quarterhourofday'] = counter
        array.append( feed.loc[feed['quarterhourofday']== counter].values)
        counter += 1

    return(feed)

def get_semester(now):
    if (now.month <= 6):
        return 0
    else:
        return 1
def get_wintermonths(now):
    if (now.month >= 4 and now.month<=9):
        return 0
    else:
        return 1

def deseasonalize(feed, r_id, grouper, plot= False):
    """
     This function deseasonalizes a time series based on a period of quarter of hour, 
     day of the week and month of the year. 
     The input is the feed to deseasonalize and its naming id.
     The process of deseasonalization is describen in [...paper...]
     It computes for each periodicity:
        1. The mean over the three aggregation periods.
        2. The normalized series by dividing it with the previous mean.
        3. The deseasonalized series dividing it by the median of the previos normalized values.
     UM_feedid is the final deseasonalized series. 
     It returns a data frame [len(feed),10] with the original feed and all the derivated series. 
    """
    
    feed = pd.DataFrame(feed)

    feed.columns = [r_id]
    
    feed = quarterofhour(feed, grouper)
    feed['dayofweek'] = feed.index.dayofweek
    feed['month'] = feed.index.month
    feed['quarter'] = feed.index.quarter
    feed['dayofyear'] = feed.index.dayofyear
    feed['weekofyear'] = feed.index.weekofyear
    feed['semester'] = [get_semester(x) for x in feed.index]
    feed['wintermonths'] = [get_wintermonths(x) for x in feed.index]
    
    
    for i in range(5):

        if i == 0:
            aggPeriod = 'D'
            granularity = '15Min'
            grouper = 'quarterhourofday'
            y = feed[r_id].values
            winSize = 15
            std = 2

        elif i == 1:
            aggPeriod = granularity = 'W'
            grouper = 'dayofweek'
            winSize = 300
            std = 100
        
#        elif i == 2:
#            aggPeriod = granularity = 'DY'
#            grouper = 'dayofyear'
#            winSize = 12000
#            std = 6000.
        
        elif i == 4:
            aggPeriod = granularity = 'WY'
            grouper = 'weekofyear'
            winSize = 1200
            std = 350
                        
        elif i == 2:
            aggPeriod = granularity = 'M'
            grouper = 'month'
            winSize = 17000
            std = 2000
        
        elif i == 3:
            aggPeriod = granularity = 'Q'
            grouper = 'quarter'
            winSize = 16000
            std = 2000

#        elif i == 4:
#            aggPeriod = granularity = 'S'
#            grouper = 'semester'
#            winSize = 8600
#            std = 4000
            

        avg = 'avg'+aggPeriod+'_'+r_id
        z = 'Z'+aggPeriod+'_'+r_id
        u = 'U'+granularity+'_'+r_id

# Period average AvgD
#        feed[avg] = feed.groupby(pd.TimeGrouper(aggPeriod)).transform('mean')[r_id]

## Moving average 1 day window
        feed.loc[:,avg] = feed[r_id].rolling(window=winSize,center=True, min_periods=1,win_type='gaussian').mean(std=std)
#
### Normalized series for 1 day z = y/avgD
        feed[z] = feed[r_id] / feed[avg]
#        
### Findind quarter of an hour means

        feed['S%s' %granularity] = feed.groupby(grouper).transform('mean')[z]

        feed[u] = y /feed['S%s' %granularity]
        
#        feed[u] = feed[u].fillna(0)
        
#        feed['S%s' %aggPeriod] = feed.groupby(grouper).transform('mean')[z]
#        feed[u] = y / feed['S%s' %aggPeriod]
        y = feed[u].values
        
        if plot:
            import matplotlib.pyplot as plt
            plt.plot(feed[r_id])
#            plt.plot(feed[avg])
#            plt.plot(feed[z])
#            plt.plot(feed['S%s' %aggPeriod])
            plt.plot(feed[u])
            plt.legend(['feed', '%s' %u])# '%s' %avg, '%s' %z, 'S%s' %aggPeriod,
            plt.show()
    
        
#    if plot:
#        import matplotlib.pyplot as plt
#        plt.plot(feed[r_id], '-', ms = 0.1)
#        plt.plot(feed[u], '-', ms= 0.1)
#        plt.plot(feed['avgD_'+r_id], '-', ms = 0.2)
#        plt.plot(feed['avgW_'+r_id], '-', ms = 0.2)
#        plt.plot(feed['avgM_'+r_id], '-', ms = 0.2)
#        plt.show()
#    print(feed)
    return(feed)
 
def tokwh(feed):
    return(feed * 10 / 3600000)

def dataQuality(feed):
    import numpy as np
    import matplotlib.pyplot as plt
    
    # This function has to be optimized ---------------------------------------     
    
    #   This retrieves the blocks of Nan in the even groups.
    block = (feed.notnull().shift(1) !=  feed.notnull()).astype(int).cumsum()
    feed = pd.concat([feed, block], axis=1)
    feed.columns = ['watts_hour','block']
    naGroups = feed.reset_index().groupby(['block'])['index'].apply(np.array)
    
    #   4. Computing statistics
    # Finding the number of groups of NaN in the series
    na_periods = np.floor(naGroups.size/2)
    
    # Finding the aumout of time the device is off for each period.
    na_time_periods = []
    for j in range(int(na_periods)):
        na_time_periods.append((naGroups[2*(j+1)].max()- naGroups[2*(j+1)].min()).astype('timedelta64[m]'))
    na_time_periods = pd.Series(na_time_periods, index = range(1, int(na_periods)*2, 2))
    
    # Finding the aumout of time the device is working for each period.
    working_time_periods = []
    for j in range(int(na_periods)):
        working_time_periods.append((naGroups[2*(j+1)-1].max()- naGroups[2*(j+1)-1].min()).astype('timedelta64[m]'))
    working_time_periods = pd.Series(working_time_periods)
    
    
    if len(working_time_periods):
        # Average time the device is off when it is off
        time_working_statistics = working_time_periods.describe()
    else:
        print('Your feed is empty.')
    
    if len(na_time_periods):
        # Average time the device is off when it is off
        time_off_statistics = na_time_periods.describe()
        # NOTE: 
        # Also proportion of time being off from total of time.
        proportion_Nas = (feed.isnull().sum() / feed.size)*100
        
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
    
def ts_to_mimo(x, window, h):
    """ Transforms time series to a format that is suitable for
        multi-input multi-output regression models.

    Arguments:
    ----------
        x:      Numpy array that contains the time series.
        window: Number of observations of the time series that
                will form one sample for the multi-input multi-output
                model.
        h:      Number of periods to forecast into the future.
    """
    import numpy as np
    
    n = int(x.shape[0])
    h = int(h)
    window = int(window)
    nobs = n - h - window + 1

    features = np.zeros((nobs, window))
    response = np.zeros((nobs, h))

    for t in range(nobs):
        features[t, :] = x[t:(t + window)]
        response[t, :] = x[(t + window):(t + window + h)]

    return features, response

def weather_to_mimo(weather, window, h):
    import numpy as np
    
    n = int(weather.shape[0])
    h = int(h)
    window = int(window)
    nobs = n - h - window + 1

    mimo = np.zeros((nobs, h))

    for t in range(nobs):
        mimo[t, :] = weather[(t + window):(t + window + h)]

    return mimo

def gridSeach(model, parameters, features, response, train, test):
    """
    This function performs a grid search over the parameter space. 
    It is simplistic and only allows certain range of values. If there
    is a parameter in the models that needs to be a list it has to be modified. 
    """
    
    import itertools
    import pandas as pd
    
    names = sorted(parameters)
    
    combinations = list(itertools.product(*(parameters[name] for name in names)))
    names.append('r2')
    model_matrix = pd.DataFrame(columns=names)

    for c in combinations:
        dictionary = dict(zip(names, c))

        model = model.set_params(**dictionary)
        
        model.fit(features[train], response[train])
        
        if 'hidden_layer_sizes' in dictionary:
            dictionary.update({'hidden_layer_sizes':[dictionary['hidden_layer_sizes']],
                               'r2':model.score(features[test], response[test])})
        else:
            dictionary.update({'r2':model.score(features[test], response[test])})
            
        model_matrix = model_matrix.append(dictionary, ignore_index=True)
        
    dictionary = dict(model_matrix.ix[model_matrix['r2'].argmax(),:-1])
    
    if 'hidden_layer_sizes' in dictionary:
        dictionary.update({'hidden_layer_sizes':dictionary['hidden_layer_sizes'][0]})
    if 'n_neighbors' in dictionary:
        dictionary.update({'n_neighbors':int(dictionary['n_neighbors'])})
    
    model = model.set_params(**dictionary)
    
    model.fit(features[train], response[train])
    
    return (model, model_matrix)

def scorer_smape(y_real, y_pred):
    """
    This function computes the symetric mean absolute percentage error and returns
    a scorer for crossvalidation.
    """
    import numpy as np
    smape = np.mean(abs(y_real-y_pred)/(y_real+y_pred), axis=0)*100
    return smape 

