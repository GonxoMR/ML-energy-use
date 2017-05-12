# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:03:20 2017



@author: Gonxo
"""
import pandas as pd
from pandas.tseries.offsets import *

def getting_saving_data(dataDir, secretsDir, apiDic, dataFile):
    """
    This function gets the data for the given apikeys and feeds. It stores the data
    and deletes the apikeys that are empty. Its output is saved in files, 
    making a local copy of the feeds and updating the apikey dictionary. 
    """
    import requests
    import numpy as np
    import pandas as pd
    import time
    types = np.unique(apiDic['type'])
    # Creating a saving file
    store = pd.HDFStore('%s\\%s.h5' %(dataDir, dataFile))
    
    # Looping for all types of feed
    for type in types:
    
        for index, row in apiDic.loc[(apiDic['type']==type),['key','id']].iterrows():
    		
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
                store['%s_%s' %(type,row['id'])] = df
    							
            else:
                print("Error: The number of elements recieved is: %i which is not multiple of 4. Decodification is not posible. Feed: %s" %(len(response.content), row['id']))

    apiDic.to_csv('%s\\apiKeyDictionary.csv' %secretsDir)

def cleanBeggining(feed):
    """
    This function deletes the initial setup problems: If measurement(t1) = 0 or
    (abs(measurements(t1)) <= 25 and measurement(t2) = 0)
            
    """
    import pandas as pd 
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
    import numpy as np 
    import pandas as pd 

#    from pandas.tseries.offsets import *
    # Deleting NaNs at the beggining and end of te series.
    feed = pd.DataFrame(feed)
    first = feed.first_valid_index()
    last = feed.last_valid_index()
    feed = feed.loc[first:last]

#   Negative values in house_consumption or grid_power are considered as errors and deleted.
    if ( r_type == 'house_consumption'):
        feed.ix[feed.ix[:,0] < 0, 'watts'] = np.nan
        
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
        
        if (r_type == 'house_consumption') or (r_type == 'solar_power'):

#           Original series
            result = pd.read_hdf(sourceFile, '%s_%s' %(r_type,r_id))
            
            result = cleaning_nans(result, r_type)
#            result = result.groupby(pd.TimeGrouper('W')).transform( lambda x: x.fillna(x.mean()))
            
      
        elif (r_type=='grid_power'):
#                print(str(r_type)+'_'+r_id)
            grid_feed = pd.read_hdf(sourceFile, '%s_%s' %(r_type,r_id))
            grid_feed = cleaning_nans(grid_feed, r_type)
#            grid_feed = grid_feed.groupby(pd.TimeGrouper('W')).transform( lambda x: x.fillna(x.mean()))
            solar_id = apiDic.ix[(apiDic['key'] == r_key) & (apiDic['type'] == 'solar_power')]['id'].values[0]
            solar_feed = pd.read_hdf(sourceFile, 'solar_power_%s' %str(solar_id))
            solar_feed = cleaning_nans(solar_feed, r_type)
#            solar_feed = solar_feed.groupby(pd.TimeGrouper('W')).transform( lambda x: x.fillna(x.mean()))
            result = grid_feed+solar_feed
        
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
    
    if period == 'quarterhourofday':
        
        counter = 0
        array = []
        for i in pd.date_range('00:00', '23:45', freq='15min'):
            HC.loc[(HC.index.hour == i.hour) & (HC.index.minute == i.minute),'quarterhourofday'] = counter
            array.append( HC.loc[HC['quarterhourofday']== counter].values)
            counter += 1
        
        return (stats.f_oneway( HC.loc[HC['quarterhourofday']==0], HC.loc[HC['quarterhourofday']==1], HC.loc[HC['quarterhourofday']==2], HC.loc[HC['quarterhourofday']==3], HC.loc[HC['quarterhourofday']==4],
                                               HC.loc[HC['quarterhourofday']==5], HC.loc[HC['quarterhourofday']==6], HC.loc[HC['quarterhourofday']==7], HC.loc[HC['quarterhourofday']==8], HC.loc[HC['quarterhourofday']==9],
                                               HC.loc[HC['quarterhourofday']==10], HC.loc[HC['quarterhourofday']==11], HC.loc[HC['quarterhourofday']==12], HC.loc[HC['quarterhourofday']==13], HC.loc[HC['quarterhourofday']==14],
                                               HC.loc[HC['quarterhourofday']==15], HC.loc[HC['quarterhourofday']==16], HC.loc[HC['quarterhourofday']==17], HC.loc[HC['quarterhourofday']==18], HC.loc[HC['quarterhourofday']==19],
                                               HC.loc[HC['quarterhourofday']==20], HC.loc[HC['quarterhourofday']==21], HC.loc[HC['quarterhourofday']==22], HC.loc[HC['quarterhourofday']==23], HC.loc[HC['quarterhourofday']==24],
                                               HC.loc[HC['quarterhourofday']==25], HC.loc[HC['quarterhourofday']==26], HC.loc[HC['quarterhourofday']==27], HC.loc[HC['quarterhourofday']==28], HC.loc[HC['quarterhourofday']==29],
                                               HC.loc[HC['quarterhourofday']==30], HC.loc[HC['quarterhourofday']==31], HC.loc[HC['quarterhourofday']==32], HC.loc[HC['quarterhourofday']==33], HC.loc[HC['quarterhourofday']==34],
                                               HC.loc[HC['quarterhourofday']==35], HC.loc[HC['quarterhourofday']==36], HC.loc[HC['quarterhourofday']==37], HC.loc[HC['quarterhourofday']==38], HC.loc[HC['quarterhourofday']==39],
                                               HC.loc[HC['quarterhourofday']==40], HC.loc[HC['quarterhourofday']==41], HC.loc[HC['quarterhourofday']==42], HC.loc[HC['quarterhourofday']==43], HC.loc[HC['quarterhourofday']==44],
                                               HC.loc[HC['quarterhourofday']==45], HC.loc[HC['quarterhourofday']==46], HC.loc[HC['quarterhourofday']==47], HC.loc[HC['quarterhourofday']==48], HC.loc[HC['quarterhourofday']==49],
                                               HC.loc[HC['quarterhourofday']==50], HC.loc[HC['quarterhourofday']==51], HC.loc[HC['quarterhourofday']==52], HC.loc[HC['quarterhourofday']==53], HC.loc[HC['quarterhourofday']==54],
                                               HC.loc[HC['quarterhourofday']==55], HC.loc[HC['quarterhourofday']==56], HC.loc[HC['quarterhourofday']==57], HC.loc[HC['quarterhourofday']==58], HC.loc[HC['quarterhourofday']==59],
                                               HC.loc[HC['quarterhourofday']==60], HC.loc[HC['quarterhourofday']==61], HC.loc[HC['quarterhourofday']==62], HC.loc[HC['quarterhourofday']==63], HC.loc[HC['quarterhourofday']==64],
                                               HC.loc[HC['quarterhourofday']==65], HC.loc[HC['quarterhourofday']==66], HC.loc[HC['quarterhourofday']==67], HC.loc[HC['quarterhourofday']==68], HC.loc[HC['quarterhourofday']==69],
                                               HC.loc[HC['quarterhourofday']==70], HC.loc[HC['quarterhourofday']==71], HC.loc[HC['quarterhourofday']==72], HC.loc[HC['quarterhourofday']==73], HC.loc[HC['quarterhourofday']==74],
                                               HC.loc[HC['quarterhourofday']==75], HC.loc[HC['quarterhourofday']==76], HC.loc[HC['quarterhourofday']==77], HC.loc[HC['quarterhourofday']==78], HC.loc[HC['quarterhourofday']==79],
                                               HC.loc[HC['quarterhourofday']==80], HC.loc[HC['quarterhourofday']==81], HC.loc[HC['quarterhourofday']==82], HC.loc[HC['quarterhourofday']==83], HC.loc[HC['quarterhourofday']==84],
                                               HC.loc[HC['quarterhourofday']==85], HC.loc[HC['quarterhourofday']==86], HC.loc[HC['quarterhourofday']==87], HC.loc[HC['quarterhourofday']==88], HC.loc[HC['quarterhourofday']==79],
                                               HC.loc[HC['quarterhourofday']==90], HC.loc[HC['quarterhourofday']==91], HC.loc[HC['quarterhourofday']==92], HC.loc[HC['quarterhourofday']==93], HC.loc[HC['quarterhourofday']==74],
                                               HC.loc[HC['quarterhourofday']==95]))
    elif period == 'dayofweek':
        
        HC['dayofweek'] = HC.index.dayofweek
        
        return (stats.f_oneway(HC.loc[HC['dayofweek']== 0], HC.loc[HC['dayofweek']== 1], 
                                               HC.loc[HC['dayofweek']== 2], HC.loc[HC['dayofweek']== 3], 
                                               HC.loc[HC['dayofweek']== 4], HC.loc[HC['dayofweek']== 5],
                                               HC.loc[HC['dayofweek']== 6]))
    elif period == 'month':
        
        HC['month'] = HC.index.month
        
        return (stats.f_oneway(HC.loc[HC['month']== 1], HC.loc[HC['month']== 2],
                                               HC.loc[HC['month']== 8],
                                               HC.loc[HC['month']== 9], HC.loc[HC['month']== 10],
                                               HC.loc[HC['month']== 11], HC.loc[HC['month']== 12]))
    
    
    else:
        print('Error: the options for period are quarterhourofday, dayofweek or month')


def quarterofhour(feed):
    feed = pd.DataFrame(feed)
    counter = 0
    array = []
    for i in pd.date_range('00:00', '23:45', freq='15min'):
        feed.loc[(feed.index.hour == i.hour) & (feed.index.minute == i.minute),'quarterhourofday'] = counter
        array.append( feed.loc[feed['quarterhourofday']== counter].values)
        counter += 1

    return(feed)


def deseasonalize(feed, r_id, plot= False):
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
    
    feed = quarterofhour(feed)
    feed['dayofweek'] = feed.index.dayofweek
    feed['month'] = feed.index.month
    for i in range(3):

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
            winSize = 200
            std = 100
                        
        elif i == 2:
            aggPeriod = granularity = 'M'
            grouper = 'month'
            winSize = 701
            std = 300
    
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
    
    model = model.set_params(**dictionary)
    
    model.fit(features[train], response[train])
    
    return (model, model_matrix)

#def test_stationarity(timeseries):
#    from statsmodels.tsa.stattools import adfuller
#    import matplotlib.pyplot as plt
#    #Determing rolling statistics
#    rolmean = pd.rolling_mean(timeseries, window=2880)
#    rolstd = pd.rolling_std(timeseries, window=2880)
#
#    #Plot rolling statistics:
#    orig = plt.plot(timeseries, color='blue',label='Original')
#    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
#    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
#    plt.legend(loc='best')
#    plt.title('Rolling Mean & Standard Deviation')
#    plt.show(block=False)
#    
#    #Perform Dickey-Fuller test:
#    print ('Results of Dickey-Fuller Test:')
#    print(timeseries.shape)
#    dftest = adfuller(timeseries, autolag='AIC')
#    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
#    for key,value in dftest[4].items():
#        dfoutput['Critical Value (%s)'%key] = value
#    print (dfoutput)
#    
#def iterative_ARIMA_fit(series, ARrange, MArange):
#    """ Iterates within the allowed values of the p and q parameters 
#    
#    Returns a dictionary with the successful fits.
#    Keys correspond to models.
#    """
#    from statsmodels.tsa.arima_model import ARIMA
#    if len(ARrange)==0:
#        ARrange = range(0, 5)
#    if len(MArange)==0:
#        MArange = range(0,5)
##    Diffrange = range(0)
#    Diff = 0
#    ARIMA_fit_results = {}
#    for AR in ARrange :
#        for MA in MArange :
##            for Diff in Diffrange:
#            model = ARIMA(series, order = (AR,Diff,MA))
#            fit_is_available = False
#            results_ARIMA = None
#            try:
#                results_ARIMA = model.fit(disp = -1, method = 'css')
#                fit_is_available = True
#            except:
#                continue
#            if fit_is_available:
#                safe_RSS = get_safe_RSS(series, results_ARIMA.fittedvalues)
#                ARIMA_fit_results[(AR,Diff,MA)]=[safe_RSS,results_ARIMA]
#    
#    return ARIMA_fit_results
#
#
#def get_safe_RSS(series, fitted_values):
#    """ Checks for missing indices in the fitted values before calculating RSS
#    
#    Missing indices are assigned as np.nan and then filled using neighboring points
#    """
#    from sklearn.metrics import r2_score
#    fitted_values_copy = fitted_values  # original fit is left untouched
#    missing_index = list(set(series.index).difference(set(fitted_values_copy.index)))
#    if missing_index:
#        nan_series = pd.Series(index = pd.to_datetime(missing_index))
#        fitted_values_copy = fitted_values_copy.append(nan_series)
#        fitted_values_copy.sort_index(inplace = True)
#        fitted_values_copy.fillna(method = 'bfill', inplace = True)  # fill holes
#        fitted_values_copy.fillna(method = 'ffill', inplace = True)
#    return r2_score(series, fitted_values_copy)
#
#def get_best_ARIMA_model_fit(series):
#    """ Returns a list with the best ARIMA model 
#    
#    The first element on the list contains the squared residual
#    The second element on the list contains the fit results
#    """
#    if t.isstationary(series)[0]:
#        ARIMA_fit_results = iterative_ARIMA_fit(series)
#        best_ARIMA = min(ARIMA_fit_results, key = ARIMA_fit_results.get)
#        return ARIMA_fit_results[best_ARIMA]