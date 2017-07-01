# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:57:10 2017

This is the main file to run the machine learning. 

It performs the following tasks:
    1 - Gets and saves the data. Both household and weather data. 
    Then for each feed:
    2 - It cleans the missing data.
    3 - It agregattes the feeds, from a granularity of 10 seconds to a granularity
    of 30 minutes. 
    4 - Takes out the seasonality of the data for periods of 30min, day of week, 
    month and quarter. 
    5 - Transforms the data form a w to kwh
    6 - Makes a data structure transformation. Form a time series of size N to 
    a matrix of time series of size N - H - window + 1
    7 - It adds the relevant time series of weather with a PCA transformation that keeps
    0.99 of the variance. 
    8 - Selects the estrategies to run.
    9 - Computes and select the different models. 
    9 - Saves the results and the models
    
@author: Gonxo
"""

# Imports
import os
#import time
import pandas as pd
import numpy as np
import mlfunctions as mlf
import featuresGenerator as ft
import strategies as stg

from sklearn.ensemble import RandomForestRegressor
#from sklearn import neighbors as knn
from sklearn import neural_network
from sklearn.metrics import r2_score, mean_absolute_error
#from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer

#----------------------- Instantiating variables -------------------------------
# Obtaining working directory
wd = os.getcwd()

# Set up the directory where the data and results will be stored
dataDir = os.path.join(wd, 'DATA_DIRECTORY')

# User Apikey and feed ids are neccesary to get the feed's data.
# Writting and reading Apikeys are needed in case there is not a reading apikey bypass.
# Direct to 'apiKeyDictionary.csv' location.
secretsDir = os.path.join(wd, 'SECRETS_DIRECTORY')
apiDic = pd.DataFrame.from_csv(os.path.join(secretsDir, 'apiKeyDictionary.csv'))
dataFile = 'raw_feed'

# Giving the parameters for 30 min aggregation---------------------------------
# As you migth know 1 day is composed by 48 periods of 30 minutes. If you have
# a different granularity you have to change this paramers accordingly. e.g.
# a 15min grouper will need a 96 h and if we are using 14 days of past data, 
# a window = 1344. 

# Granularity after aggregation
grouper = '30min'
# Window in the past (14 days) to use as time series. It is proved that bigger windows
# improve accuracy but they also increase the computation time. 
window = 672
# Horizon in the future to forecast. We are forecasting 1 day ahead that is 48 30min periods. 
h = 48

# Set these variables as true to download feeds and weather data. 
# If you already have downloaded the data set to False.
fetch_data = False
fetch_weather_data = False

# Objects to store the results
results_path = os.path.join(dataDir, 'RESULTS', '%s_3.csv' %grouper)
forecast_path = os.path.join(dataDir, 'FORECAST', '%s_3.csv' %grouper)
columns = ['feed', 'type', 'strategy', 'model', 'measure', 'time']

for i  in range(h):
    columns.append('t_%i' %(i+1))

results = pd.DataFrame(columns=columns)
predictions = pd.DataFrame(columns=columns)
del predictions['measure']

# Creating a measure object to evaluate the models. 
smape = make_scorer(mlf.scorer_smape, greater_is_better=False)


# ----------------------- Downloading the data --------------------------------
# Storing destination ./dataDir/dataFile.h5 
if fetch_data:
    mlf.getting_saving_data(dataDir, secretsDir, apiDic, dataFile)
    apiDic = pd.read_csv(os.path.join(secretsDir, 'apiKeyDictionary.csv'),
                         sep=None, engine='python')
if fetch_weather_data:
    runfile(os.path.join(wd, 'obtaining_weather_data.py'), wdir=wd)

# Perfroming the algorithm steps 2 - 9 for each feed. 
for index, row in apiDic.ix[:, ['key', 'type', 'id']].iterrows():

    # Local variables instantiation
    r_type = str(row['type'])
    r_id = str(row['id'])
    r_key = str(row['key'])

    # Steps 2 & 3 -------------------------------------------------------------
    # Nas cleaning takes place in the next step.
    # Solar or Solar + Grid with Nas cleaned.
    raw_feed = mlf.aggregation(r_key, r_type, r_id,
                               os.path.join(dataDir, '%s.h5' %dataFile), apiDic)
    
    feed = raw_feed.groupby(pd.TimeGrouper(freq=grouper)).sum()

    # 4 Deseasonalize data ---------------------------------------------------- 
    feed = mlf.deseasonalize(feed.ix[1:], r_id, grouper, plot=False)

    # 5 Transform into kwh ----------------------------------------------------
    feed['kwh'] = mlf.tokwh(feed.ix[:, -1])

# I thought on the necesity of normalization. Transformation into kwh is good
# enougth for household volumnes. I might delete this later. 
#    feed['normalized'] = ((feed.ix[:,-1] - feed.ix[:,-1].min()) /
                         #(feed.ix[:,-1].max() - feed.ix[:,-1].min()))

# Creating features, joining with weather data.
    # Steps 6 and 7 take place within this function. --------------------------
    features, response = ft.featureCreation(feed.ix[feed.ix[:, -1].first_valid_index():, -1],
                                            window, h, grouper,
                                            dataDir, apiDic, r_id)

    # Dividing the data into train, test and validation datasets. As feedbak in 
    # my thesis they said that validate against other houses might be worht to study.
    train = range(int(features.shape[0]*0.7))
    test = range(int(features.shape[0]*0.7), int(features.shape[0]*0.9))
    validation = range(int(features.shape[0]*0.9), features.shape[0])

    # Step 8 training models. I will give coments in the interesting parts. ------
    # K-nearest neighbours was also included in previous tries but the performance
    # was really bad.
    # I want to try SVR with multi output.
    for model in ['ANN', 'RandFor']:
        
        if model == 'ANN':
            # Setting parameters for the ANN. Only 2 network architectures are
            # tested. Automatizing architecture selection is one of the priorities.
            parameters = {'hidden_layer_sizes':[(600, 300), (600, 300, 100)],
                          'activation':['logistic', 'relu'],
                          'alpha':[1e-5, 1e-4, 1e-3], 'tol':[1e-4, 1e-5]}
            to_run = neural_network.MLPRegressor(learning_rate_init=0.0001,
                                                 verbose=False, batch_size=4000)
        
        if model == 'RandFor':
            parameters = {'max_depth':[30, 50, 70]}
            to_run = RandomForestRegressor(n_jobs=-1)
        
        # Performing a seach to find the best hyperparameters
        clf = RandomizedSearchCV(to_run, param_distributions=parameters,
                                 scoring=smape, n_iter=7, cv=[(train, test)])
        
        # This selects the different strategies to run. Each strategy is defined
        # by two parameters, the horizon to forecast and the size of each one of
        # the individual forecasts. The format to be given is [horizon, size].
        # go to ../strategies.py for more info. 
        # Some strategies:
            # [h,1] Direct strategy
            # [h,6] DIRMO strategy with 8 chunks of forecast of size 6. 
            # [h,h] MIMO strategy. 
        for horizon, size in [[h, 1], [h, 6], [h, h]]:

            strategy = stg.findStrategy(horizon, size)

            models = stg.train_forecast_strategies(features, response, clf,
                                                   horizon=horizon, size=size)

            predict = stg.forecast(features[validation, :], models=models,
                                   horizon=horizon, size=size,
                                   strategy=strategy)

            r2 = ['R2']
            mae = ['MAE']
            mape = ['MAPE']
            smape = ['SMAPE']
            for i in range(int(horizon/size)):
                for measure in ['R2', 'MAE', 'MAPE', 'SMAPE']:
                    if measure == 'R2':
                        r2.extend(r2_score(response[validation, i],
                                           predict, multioutput='raw_values'))
                    if measure == 'MAE':
                        mae.extend(mean_absolute_error(response[validation, i],
                                                       predict, multioutput='raw_values'))
                    if measure == 'MAPE':
                        mape.extend(np.mean((np.abs(response[validation, i] - predict.transpose())/ response[validation, i])*100, axis=1))
                    if measure == 'SMAPE':
                        smape.extend(mlf.scorer_smape(response[validation, i], predict))

            c = ['measure']
            c.extend(['t_%i' %(i+1) for i in range(horizon)])
            r = pd.DataFrame([r2, mae, mape, smape], columns=c)

            # Step 9 ----------------------------------------------------------
            r['feed'] = r_id
            r['strategy'] = strategy
            r['model'] = model
            r['type'] = r_type
            results = results.append(r)
            results.to_csv(results_path)

            predict['feed'] = r_id
            predict['strategy'] = strategy
            predict['model'] = model
            predict['type'] = r_type
            predictions = predictions.append(predict)
            predictions.to_csv(forecast_path)



# --------------------- Deprecated. to delete soon ----------------------------
#
#    for strategy in ['MIMO', 'REC', 'DIR', 'DIRMO']:
#
#        if strategy == 'MIMO':
#
#        # NN, KNN, RandForest. (SVR)
#            for j in range(3):
#                if j == 0:
#                    parameters = {'hidden_layer_sizes':[(600, 300), (600, 300, 100)],
#                                  'activation':['logistic', 'relu'],
#                                  'alpha':[1e-5, 1e-4, 1e-3], 'tol':[1e-4, 1e-5]}
#                    nn = neural_network.MLPRegressor(learning_rate_init=0.0001,
#                                                     verbose=False, batch_size=4000)
#                    model = 'NN'
#                    n_iter_search = 7
#                    clf = RandomizedSearchCV(nn, param_distributions=parameters,
#                                             n_iter=n_iter_search, cv=[(train, test)])
#                    print('Ready to run %s_%s_%s_%s' %(model, r_type, r_id, grouper))
#                    start = time.time()
#                    clf.fit(features, response)
#                    running_time = time.time() - start
#                    print(running_time)
#
#                if j == 1:
#                    parameters = {'max_depth':[30, 50, 70]}
#                    clf = RandomForestRegressor(n_jobs=-1)
#                    model = 'RandFor'
#                    print('Ready to run %s_%s_%s_%s' %(model, r_type, r_id, grouper))
#                    start = time.time()
#
#                    clf, model_matrix = mlf.gridSeach(clf, parameters, features,
#                                                      response, train, test)
#
#                    running_time = time.time() - start
#                    print(running_time)
#
#                if j == 2:
#                    model = 'KNNR'
#                    parameters = {'n_neighbors':[5, 20, 50, 100],
#                                  'weights':('uniform', 'distance')}
#                    clf = knn.KNeighborsRegressor(n_jobs=-1)
#
#                    print('Ready to run %s_%s_%s_%s' %(model, r_type,
#                                                       r_id, grouper))
#                    start = time.time()
#
#                    clf, model_matrix = mlf.gridSeach(clf, parameters, features,
#                                                      response, train, test)
#
#                    running_time = time.time() - start
#                    print(running_time)
#
#                joblib.dump(clf, os.path.join(wd, 'MODELS',
#                                              '%s_%s_%s_%s.pkl' %(model, r_type,
#                                                                  r_id, grouper)))
#
#                prediction = pd.DataFrame(clf.predict(features[validation]),
#                                          columns=['t_%i' %(i+1) for i in range(h)])
#
#                for measure in ['R2', 'MAE', 'MAPE']:
#                    if measure == 'R2':
#                        scores = r2_score(response[validation], prediction,
#                                          multioutput='raw_values')
#                    if measure == 'MAE':
#                        scores = mean_absolute_error(response[validation],
#                                                     prediction, multioutput='raw_values')
#                    if measure == 'MAPE':
#                        scores = np.mean((np.abs(response[validation] - prediction)/ response[validation])*100, axis=0)
#
#                    c = [r_id, r_type, strategy, model, measure, running_time]
#                    c.extend(scores)
#
#                    results = results.append(pd.DataFrame(c, index=columns).transpose())
#                    results.to_csv(results_path)
#
#                prediction['feed'] = r_id
#                prediction['strategy'] = strategy
#                prediction['model'] = model
#                prediction['type'] = r_type
#                predictions = predictions.append(prediction)
#                predictions.to_csv(forecast_path)
#
#                del (c, clf, prediction, scores)
#
#        if strategy == 'DIR':
#
#            for j in range(3):
#
#                if j == 0:
#                    parameters = {'hidden_layer_sizes':[(600, 300), (600, 300, 100)],
#                                  'activation':['logistic', 'relu'],
#                                  'alpha':[1e-3, 1e-4, 1e-5], 'tol':[1e-4, 1e-5]}
#                    nn = neural_network.MLPRegressor(learning_rate_init=0.0001,
#                                                     verbose=False, batch_size=4000)
#                    model = 'NN'
#                    n_iter_search = 7
#                    clf = RandomizedSearchCV(nn, param_distributions=parameters,
#                                             n_iter=n_iter_search, cv=[(train, test)])
#                if j == 1:
#                    parameters = {'max_depth': [30, 50, 70]}
#                    clf = RandomForestRegressor(n_jobs=-1)
#                    model = 'RandFor'
#                if j == 2:
#                    model = 'KNNR'
#                    parameters = {'n_neighbors':[5, 20, 50, 100],
#                                  'weights':('uniform', 'distance')}
#                    clf = knn.KNeighborsRegressor(n_jobs=-1)
#
#                predict = pd.DataFrame()
#                r2 = ['R2']
#                mae = ['MAE']
#                mape = ['MAPE']
#                start = time.time()
#                for i in range(h):
#                    print('Ready to run %s_%s_%s_t%d' %(model, r_type, r_id, (i+1)))
#                    if model == 'NN':
#                        clf.fit(features, response[:, i])
#
#                    else:
#                        clf, model_matrix = mlf.gridSeach(clf, parameters,
#                                                          features, response[:, i],
#                                                          train, test)
#
##                        clf.fit(features[train],response[train,i])
#                    # print(clf.score(features[test],response[test]))
#                    joblib.dump(clf, os.path.join(wd, 'MODELS',
#                                                  '%s_%s_%s_%s_t%d.pkl' %(model, r_type,
#                                                                          r_id, grouper, (i+1))))
#
#                    prediction = pd.DataFrame(clf.predict(features[validation]),
#                                              columns=['t_%i'%(i+1)])
#
#                    predict = pd.concat([predict, prediction], axis=1)
#
#                    for measure in ['R2', 'MAE', 'MAPE']:
#                        if measure == 'R2':
#                            r2.extend(r2_score(response[validation, i],
#                                               prediction, multioutput='raw_values'))
#                        if measure == 'MAE':
#                            mae.extend(mean_absolute_error(response[validation, i],
#                                                           prediction, multioutput='raw_values'))
#                        if measure == 'MAPE':
#                            mape.extend(np.mean((np.abs(response[validation, i] - prediction.transpose())/ response[validation, i])*100, axis=1))
#
#                running_time = time.time() - start
#                print(running_time)
#                c = ['measure']
#                c.extend(['t_%i' %(i+1) for i in range(h)])
#                r = pd.DataFrame([r2, mae, mape], columns=c)
#
#                r['feed'] = r_id
#                r['strategy'] = strategy
#                r['model'] = model
#                r['type'] = r_type
#                r['time'] = running_time
#                results = results.append(r)
#
#                results.to_csv(results_path)
#
#                predict['feed'] = r_id
#                predict['strategy'] = strategy
#                predict['model'] = model
#                predict['type'] = r_type
#                predictions = predictions.append(predict)
#                predictions.to_csv(forecast_path)
#
#                del (c, clf, r2, mae, mape, r)
#
#        if strategy == 'REC':
#
#            predRec = range(int(features.shape[0]*0.7), int(features.shape[0]))
#            for j in range(3):
#
#                if j == 0:
#                    parameters = {'hidden_layer_sizes':[(600, 300), (600, 300, 100)],
#                                  'activation': ('logistic', 'relu'),
#                                  'alpha':[1e-3, 1e-4, 1e-5], 'tol':[1e-4, 1e-5]}
#                    nn = neural_network.MLPRegressor(learning_rate_init=0.0001,
#                                                     verbose=False, batch_size=4000)
#                    model = 'NN'
#                    n_iter_search = 7
#                    clf = RandomizedSearchCV(nn, param_distributions=parameters,
#                                             n_iter=n_iter_search, cv=[(train, test)])
#                if j == 1:
#                    parameters = {'max_depth': [30, 50, 70]}
#                    clf = RandomForestRegressor(n_jobs=-1)
#                    model = 'RandFor'
#                if j == 2:
#                    model = 'KNNR'
#                    parameters = {'n_neighbors':[5, 20, 50, 100],
#                                  'weights':('uniform', 'distance')}
#                    clf = knn.KNeighborsRegressor(n_jobs=-1)
#
#                r2 = ['R2']
#                mae = ['MAE']
#                mape = ['MAPE']
#
#                print('Ready to run %s_%s_%s_t%d' %(model, r_type, r_id, (1)))
#                start = time.time()
#
#                if model == 'NN':
#                    clf.fit(features, response[:, 0])
#                else:
#
#                    clf, model_matrix = mlf.gridSeach(clf, parameters, features,
#                                                      response[:, 0], train, test)
#
#                joblib.dump(clf, os.path.join(wd, 'MODELS',
#                                              '%s_%s_%s_%s_t1.pkl' %(model, r_type, r_id, grouper)))
#
#                pd.DataFrame(np.zeros((predRec[-1]-predRec[0], h)))
#                feat = pd.DataFrame(features[:])
#                for i in range(h):
#                    print('predicting %s' %(i+1))
#
#                    predict.ix[i:,i] = clf.predict(feat.ix[i:, :])[:-1]
#                    feat.ix[(i+1):, (feat.shape[1]-1-i):] = predict.ix[i:,:i].values
#
#                predict = predict.ix[(validation[0]-predRec[0]-1):,:]
#                predict.columns = ['t_%i' %(i+1) for i in range(h)]
#
#                running_time = time.time() - start
#                print(running_time)
#
#                del clf, feat
#
#                for measure in ['R2', 'MAE', 'MAPE']:
#
#                    if measure == 'R2':
#                        r2.extend(r2_score(response[validation[h+1:]], predict,
#                                           multioutput='raw_values'))
#                    if measure == 'MAE':
#                        mae.extend(mean_absolute_error(response[validation[h+1:]],
#                                                       predict, multioutput='raw_values'))
#                    if measure == 'MAPE':
#                        mape.extend(np.mean((np.abs(response[validation[h+1:]] - predict)/ response[validation[h+1:]])*100, axis=0))
#
#                c = ['measure']
#                c.extend(['t_%i' %(i+1) for i in range(h)])
#
#                r = pd.DataFrame([r2, mae, mape], columns=c)
#
#                r['feed'] = r_id
#                r['strategy'] = strategy
#                r['model'] = model
#                r['type'] = r_type
#                r['time'] = running_time
#                results = results.append(r)
#
#                results.to_csv(results_path)
#
#                predict['feed'] = r_id
#                predict['strategy'] = strategy
#                predict['model'] = model
#                predict['type'] = r_type
#                predictions = predictions.append(predict.ix[:, :])
#                predictions.to_csv(forecast_path)
#                del  mae, mape, r2, r
#
#        if strategy == 'DIRMO':
#
#            len_models = [6]
#            for j in range(3):
#
#                if j == 0:
#                    parameters = {'hidden_layer_sizes':[(600, 300), (600, 300, 100)],
#                                  'activation': ('logistic', 'relu'),
#                                  'alpha':[1e-3, 1e-4, 1e-5], 'tol':[1e-4, 1e-5]}
#                    nn = neural_network.MLPRegressor(learning_rate_init=0.0001,
#                                                     verbose=False, batch_size=4000)
#                    model = 'NN'
#                    n_iter_search = 7
#                    clf = RandomizedSearchCV(nn, param_distributions=parameters,
#                                             n_iter=n_iter_search,
#                                             cv=[(train, test)])
#                if j == 1:
#                    parameters = {'max_depth':[30, 50, 70]}
#                    clf = RandomForestRegressor(n_jobs=-1)
#                    model = 'RandFor'
#                if j == 2:
#                    model = 'KNNR'
#                    parameters = {'n_neighbors':[5, 20, 50, 100],
#                                  'weights': ('uniform', 'distance')}
#                    clf = knn.KNeighborsRegressor(n_jobs=-1)
#
#                predict = pd.DataFrame()
#                r2 = ['R2']
#                mae = ['MAE']
#                mape = ['MAPE']
#                start = time.time()
#
#                for s in len_models:
#                    n = int(h/s)
#                    for i in range(n):
#                        print(range((i*s), ((i+1)*s)))
#                        print('Ready to run %s_%s_%s_t%d' %(model, r_type,
#                                                            r_id, ((i+1)*s)))
#
#                        if model == 'NN':
#                            clf.fit(features, response[:, (i*s):((i+1)*s)])
#                        else:
#                            clf, model_matrix = mlf.gridSeach(clf, parameters,
#                                                              features,
#                                                              response[:, (i*s):((i+1)*s)],
#                                                              train, test)
#
#                        joblib.dump(clf, os.path.join(wd, 'MODELS',
#                                                      '%s_%s_%s_%s_dirmo_t%d.pkl' %(model, r_type,
#                                                                                    r_id, grouper, (i*s)+1)))
#
#                        prediction = pd.DataFrame(clf.predict(features[validation]),
#                                                  columns=['t_%i'%(j+1) for j in range((i*s), ((i+1)*s))])
#
#                        predict = pd.concat([predict, prediction], axis=1)
#
#                        for measure in ['R2', 'MAE', 'MAPE']:
#                            if measure == 'R2':
#                                r2.extend(r2_score(response[validation, (i*s):((i+1)*s)],
#                                                   prediction, multioutput='raw_values'))
#                            if measure == 'MAE':
#                                mae.extend(mean_absolute_error(response[validation, (i*s):((i+1)*s)],
#                                                               prediction, multioutput='raw_values'))
#                            if measure == 'MAPE':
#                                mape.extend(np.mean((np.abs(response[validation, (i*s):((i+1)*s)] - prediction)/ response[validation, (i*s):((i+1)*s)]), axis=0)*100)
#
#                    running_time = time.time() - start
#                    print(running_time/60)
#                    c = ['measure']
#                    c.extend(['t_%i' %(i+1) for i in range(h)])
#                    r = pd.DataFrame([r2, mae, mape], columns=c)
#
#                    r['feed'] = r_id
#                    r['strategy'] = strategy
#                    r['model'] = model
#                    r['type'] = r_type
#                    r['time'] = running_time
#                    results = results.append(r)
#
#                    results.to_csv(results_path)
#
#                    predict['feed'] = r_id
#                    predict['strategy'] = strategy
#                    predict['model'] = model
#                    predict['type'] = r_type
#                    predictions = predictions.append(predict)
#
#                    predictions.to_csv(forecast_path)
#
#                    del (c, r2, mae, mape, r)
#        # Compound Model
#        # Un normalize
#        # Re-seasonalize
#        # Final forecast
        