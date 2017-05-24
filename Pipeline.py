# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:57:10 2017

@author: Gonxo
"""
import os
import time
import pandas as pd
import numpy as np
import mlfunctions as mlf
import featuresGenerator as ft

from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors as knn
from sklearn import neural_network
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV

# Loading Data
# Obtaining working directory
wd = os.getcwd()
# Set up the destination  and secrects directory
# Set up the destination directory
dataDir = os.path.join(wd, 'DATA_DIRECTORY')

# User Apikey and feed ids are neccesary to get the feed's data.
# Writting and reading Apikeys are needed in case there is not a reading apikey bypass.
# Direct to 'apiKeyDictionary.csv' location.
secretsDir = os.path.join(wd, 'SECRETS_DIRECTORY')
apiDic = pd.DataFrame.from_csv(os.path.join(secretsDir, 'apiKeyDictionary.csv'))
dataFile = 'raw_feed'
grouper = '30min'

# If you already have downloaded the data set to False.
fetch_data = False
fetch_weather_data = False

if fetch_data:
    mlf.getting_saving_data(dataDir, secretsDir, apiDic, dataFile)
    apiDic = pd.read_csv(os.path.join(secretsDir, 'apiKeyDictionary.csv'),
                         sep=None, engine='python')
if fetch_weather_data:
    runfile(os.path.join(wd, 'obtaining_weather_data.py'), wdir=wd)

# Group by 15 min and 30 min

#for grouper in [ '30min','15min']:
#    window = 672
#    h = 96
#if grouper == '30min':
window = 672
h = 48

# Saving Results object
results_path = os.path.join(dataDir, 'RESULTS', '%s_3.csv' %grouper)
forecast_path = os.path.join(dataDir, 'FORECAST', '%s_3.csv' %grouper)
columns = ['feed', 'type', 'strategy', 'model', 'measure', 'time']

for i  in range(h):
    columns.append('t_%i' %(i+1))

results = pd.DataFrame(columns=columns)
predictions = pd.DataFrame(columns=columns)
del predictions['measure']

# This is new
for index, row in apiDic.ix[[1], ['key', 'type', 'id']].iterrows():

    r_type = str(row['type'])
    r_id = str(row['id'])
    r_key = str(row['key'])
    print(r_id)

    # Nas cleaning takes place in the next step.
    # Solar or Solar + Grid with Nas cleaned.

    raw_feed = mlf.aggregation(r_key, r_type, r_id,
                               os.path.join(dataDir, '%s.h5' %dataFile), apiDic)
    feed = raw_feed.groupby(pd.TimeGrouper(freq=grouper)).sum()

    # Deseasonalize data
    feed = mlf.deseasonalize(feed.ix[1:], r_id, grouper, plot=False)

    # Normalize data
    feed['kwh'] = mlf.tokwh(feed.ix[:, -1])
#    feed.ix[:,'kwh'].plot()
#    feed['normalized'] = ((feed.ix[:,-1] - feed.ix[:,-1].min()) /
                         #(feed.ix[:,-1].max() - feed.ix[:,-1].min()))

        # Creating features, joining with weather data.

    features, response = ft.featureCreation(feed.ix[feed.ix[:, -1].first_valid_index():, -1],
                                            window, h, grouper,
                                            dataDir, apiDic, r_id)

    train = range(int(features.shape[0]*0.7))
    test = range(int(features.shape[0]*0.7), int(features.shape[0]*0.9))
    validation = range(int(features.shape[0]*0.9), features.shape[0])

    # MIMO, DIR and Recursive
    if r_id == '3009':
        strategies = ['DIRMO']
    else:
        strategies = ['MIMO', 'REC', 'DIR', 'DIRMO']
        
    for strategy in strategies:

        if strategy == 'MIMO':

        # NN, KNN, RandForest. (SVR)
            for j in range(3):
                if j == 0:
                    parameters = {'hidden_layer_sizes':[(600, 300), (600, 300, 100)],
                                  'activation':['logistic', 'relu'],
                                  'alpha':[1e-5, 1e-4, 1e-3], 'tol':[1e-4, 1e-5]}
                    nn = neural_network.MLPRegressor(learning_rate_init=0.0001,
                                                     verbose=False, batch_size=4000)
                    model = 'NN'
                    n_iter_search = 7
                    clf = RandomizedSearchCV(nn, param_distributions=parameters,
                                             n_iter=n_iter_search, cv=[(train, test)])
                    print('Ready to run %s_%s_%s_%s' %(model, r_type, r_id, grouper))
                    start = time.time()
                    clf.fit(features, response)
                    running_time = time.time() - start
                    print(running_time)

                if j == 1:
                    parameters = {'max_depth':[30, 50, 70]}
                    clf = RandomForestRegressor(n_jobs=-1)
                    model = 'RandFor'
                    print('Ready to run %s_%s_%s_%s' %(model, r_type, r_id, grouper))
                    start = time.time()

                    clf, model_matrix = mlf.gridSeach(clf, parameters, features,
                                                      response, train, test)

                    running_time = time.time() - start
                    print(running_time)

                if j == 2:
                    model = 'KNNR'
                    parameters = {'n_neighbors':[5, 20, 50, 100],
                                  'weights':('uniform', 'distance')}
                    clf = knn.KNeighborsRegressor(n_jobs=-1)

                    print('Ready to run %s_%s_%s_%s' %(model, r_type,
                                                       r_id, grouper))
                    start = time.time()

                    clf, model_matrix = mlf.gridSeach(clf, parameters, features,
                                                      response, train, test)

                    running_time = time.time() - start
                    print(running_time)

                joblib.dump(clf, os.path.join(wd, 'MODELS',
                                              '%s_%s_%s_%s.pkl' %(model, r_type,
                                                                  r_id, grouper)))

                prediction = pd.DataFrame(clf.predict(features[validation]),
                                          columns=['t_%i' %(i+1) for i in range(h)])

                for measure in ['R2', 'MAE', 'MAPE']:
                    if measure == 'R2':
                        scores = r2_score(response[validation], prediction,
                                          multioutput='raw_values')
                    if measure == 'MAE':
                        scores = mean_absolute_error(response[validation],
                                                     prediction, multioutput='raw_values')
                    if measure == 'MAPE':
                        scores = np.mean((np.abs(response[validation] - prediction)/ response[validation])*100, axis=0)

                    c = [r_id, r_type, strategy, model, measure, running_time]
                    c.extend(scores)

                    results = results.append(pd.DataFrame(c, index=columns).transpose())
                    results.to_csv(results_path)

                prediction['feed'] = r_id
                prediction['strategy'] = strategy
                prediction['model'] = model
                prediction['type'] = r_type
                predictions = predictions.append(prediction)
                predictions.to_csv(forecast_path)

                del (c, clf, prediction, scores)

        if strategy == 'DIR':

            for j in range(3):

                if j == 0:
                    parameters = {'hidden_layer_sizes':[(600, 300), (600, 300, 100)],
                                  'activation':['logistic', 'relu'],
                                  'alpha':[1e-3, 1e-4, 1e-5], 'tol':[1e-4, 1e-5]}
                    nn = neural_network.MLPRegressor(learning_rate_init=0.0001,
                                                     verbose=False, batch_size=4000)
                    model = 'NN'
                    n_iter_search = 7
                    clf = RandomizedSearchCV(nn, param_distributions=parameters,
                                             n_iter=n_iter_search, cv=[(train, test)])
                if j == 1:
                    parameters = {'max_depth': [30, 50, 70]}
                    clf = RandomForestRegressor(n_jobs=-1)
                    model = 'RandFor'
                if j == 2:
                    model = 'KNNR'
                    parameters = {'n_neighbors':[5, 20, 50, 100],
                                  'weights':('uniform', 'distance')}
                    clf = knn.KNeighborsRegressor(n_jobs=-1)

                predict = pd.DataFrame()
                r2 = ['R2']
                mae = ['MAE']
                mape = ['MAPE']
                start = time.time()
                for i in range(h):
                    print('Ready to run %s_%s_%s_t%d' %(model, r_type, r_id, (i+1)))
                    if model == 'NN':
                        clf.fit(features, response[:, i])

                    else:
                        clf, model_matrix = mlf.gridSeach(clf, parameters,
                                                          features, response[:, i],
                                                          train, test)

#                        clf.fit(features[train],response[train,i])
                    # print(clf.score(features[test],response[test]))
                    joblib.dump(clf, os.path.join(wd, 'MODELS',
                                                  '%s_%s_%s_%s_t%d.pkl' %(model, r_type,
                                                                          r_id, grouper, (i+1))))

                    prediction = pd.DataFrame(clf.predict(features[validation]),
                                              columns=['t_%i'%(i+1)])

                    predict = pd.concat([predict, prediction], axis=1)

                    for measure in ['R2', 'MAE', 'MAPE']:
                        if measure == 'R2':
                            r2.extend(r2_score(response[validation, i],
                                               prediction, multioutput='raw_values'))
                        if measure == 'MAE':
                            mae.extend(mean_absolute_error(response[validation, i],
                                                           prediction, multioutput='raw_values'))
                        if measure == 'MAPE':
                            mape.extend(np.mean((np.abs(response[validation, i] - prediction.transpose())/ response[validation, i])*100, axis=1))

                running_time = time.time() - start
                print(running_time)
                c = ['measure']
                c.extend(['t_%i' %(i+1) for i in range(h)])
                r = pd.DataFrame([r2, mae, mape], columns=c)

                r['feed'] = r_id
                r['strategy'] = strategy
                r['model'] = model
                r['type'] = r_type
                r['time'] = running_time
                results = results.append(r)

                results.to_csv(results_path)

                predict['feed'] = r_id
                predict['strategy'] = strategy
                predict['model'] = model
                predict['type'] = r_type
                predictions = predictions.append(predict)
                predictions.to_csv(forecast_path)

                del (c, clf, r2, mae, mape, r)

        if strategy == 'REC':
            
            predRec = range(int(features.shape[0]*0.7), int(features.shape[0]))
            for j in range(3):

                if j == 0:
                    parameters = {'hidden_layer_sizes':[(600, 300), (600, 300, 100)],
                                  'activation': ('logistic', 'relu'),
                                  'alpha':[1e-3, 1e-4, 1e-5], 'tol':[1e-4, 1e-5]}
                    nn = neural_network.MLPRegressor(learning_rate_init=0.0001,
                                                     verbose=False, batch_size=4000)
                    model = 'NN'
                    n_iter_search = 7
                    clf = RandomizedSearchCV(nn, param_distributions=parameters,
                                             n_iter=n_iter_search, cv=[(train, test)])
                if j == 1:
                    parameters = {'max_depth': [30, 50, 70]}
                    clf = RandomForestRegressor(n_jobs=-1)
                    model = 'RandFor'
                if j == 2:
                    model = 'KNNR'
                    parameters = {'n_neighbors':[5, 20, 50, 100],
                                  'weights':('uniform', 'distance')}
                    clf = knn.KNeighborsRegressor(n_jobs=-1)

                r2 = ['R2']
                mae = ['MAE']
                mape = ['MAPE']

                print('Ready to run %s_%s_%s_t%d' %(model, r_type, r_id, (1)))
                start = time.time()

                if model == 'NN':
                    clf.fit(features, response[:, 0])
                else:

                    clf, model_matrix = mlf.gridSeach(clf, parameters, features,
                                                      response[:, 0], train, test)

                joblib.dump(clf, os.path.join(wd, 'MODELS',
                                              '%s_%s_%s_%s_t1.pkl' %(model, r_type, r_id, grouper)))

                pd.DataFrame(np.zeros((predRec[-1]-predRec[0], h)))
                feat = pd.DataFrame(features[:])
                for i in range(h):
                    print('predicting %s' %(i+1))

                    predict.ix[i:,i] = clf.predict(feat.ix[i:, :])[:-1]
                    feat.ix[(i+1):, (feat.shape[1]-1-i):] = predict.ix[i:,:i].values
                
                predict = predict.ix[(validation[0]-predRec[0]-1):,:]
                predict.columns = ['t_%i' %(i+1) for i in range(h)]

                running_time = time.time() - start
                print(running_time)

                del clf, feat

                for measure in ['R2', 'MAE', 'MAPE']:

                    if measure == 'R2':
                        r2.extend(r2_score(response[validation[h+1:]], predict,
                                           multioutput='raw_values'))
                    if measure == 'MAE':
                        mae.extend(mean_absolute_error(response[validation[h+1:]],
                                                       predict, multioutput='raw_values'))
                    if measure == 'MAPE':
                        mape.extend(np.mean((np.abs(response[validation[h+1:]] - predict)/ response[validation[h+1:]])*100, axis=0))

                c = ['measure']
                c.extend(['t_%i' %(i+1) for i in range(h)])

                r = pd.DataFrame([r2, mae, mape], columns=c)

                r['feed'] = r_id
                r['strategy'] = strategy
                r['model'] = model
                r['type'] = r_type
                r['time'] = running_time
                results = results.append(r)

                results.to_csv(results_path)

                predict['feed'] = r_id
                predict['strategy'] = strategy
                predict['model'] = model
                predict['type'] = r_type
                predictions = predictions.append(predict.ix[:, :])
                predictions.to_csv(forecast_path)
                del  mae, mape, r2, r

        if strategy == 'DIRMO':

            len_models = [6]
            for j in range(3):

                if j == 0:
                    parameters = {'hidden_layer_sizes':[(600, 300), (600, 300, 100)],
                                  'activation': ('logistic', 'relu'),
                                  'alpha':[1e-3, 1e-4, 1e-5], 'tol':[1e-4, 1e-5]}
                    nn = neural_network.MLPRegressor(learning_rate_init=0.0001,
                                                     verbose=False, batch_size=4000)
                    model = 'NN'
                    n_iter_search = 7
                    clf = RandomizedSearchCV(nn, param_distributions=parameters,
                                             n_iter=n_iter_search,
                                             cv=[(train, test)])
                if j == 1:
                    parameters = {'max_depth':[30, 50, 70]}
                    clf = RandomForestRegressor(n_jobs=-1)
                    model = 'RandFor'
                if j == 2:
                    model = 'KNNR'
                    parameters = {'n_neighbors':[5, 20, 50, 100],
                                  'weights': ('uniform', 'distance')}
                    clf = knn.KNeighborsRegressor(n_jobs=-1)

                predict = pd.DataFrame()
                r2 = ['R2']
                mae = ['MAE']
                mape = ['MAPE']
                start = time.time()

                for s in len_models:
                    n = int(h/s)
                    for i in range(n):
                        print(range((i*s), ((i+1)*s)))
                        print('Ready to run %s_%s_%s_t%d' %(model, r_type,
                                                            r_id, ((i+1)*s)))

                        if model == 'NN':
                            clf.fit(features, response[:, (i*s):((i+1)*s)])
                        else:
                            clf, model_matrix = mlf.gridSeach(clf, parameters,
                                                              features,
                                                              response[:, (i*s):((i+1)*s)],
                                                              train, test)

                        joblib.dump(clf, os.path.join(wd, 'MODELS',
                                                      '%s_%s_%s_%s_dirmo_t%d.pkl' %(model, r_type,
                                                                                    r_id, grouper, (i*s)+1)))

                        prediction = pd.DataFrame(clf.predict(features[validation]),
                                                  columns=['t_%i'%(j+1) for j in range((i*s), ((i+1)*s))])

                        predict = pd.concat([predict, prediction], axis=1)

                        for measure in ['R2', 'MAE', 'MAPE']:
                            if measure == 'R2':
                                r2.extend(r2_score(response[validation, (i*s):((i+1)*s)],
                                                   prediction, multioutput='raw_values'))
                            if measure == 'MAE':
                                mae.extend(mean_absolute_error(response[validation, (i*s):((i+1)*s)],
                                                               prediction, multioutput='raw_values'))
                            if measure == 'MAPE':
                                mape.extend(np.mean((np.abs(response[validation, (i*s):((i+1)*s)] - prediction)/ response[validation, (i*s):((i+1)*s)]), axis=0)*100)

                    running_time = time.time() - start
                    print(running_time/60)
                    c = ['measure']
                    c.extend(['t_%i' %(i+1) for i in range(h)])
                    r = pd.DataFrame([r2, mae, mape], columns=c)

                    r['feed'] = r_id
                    r['strategy'] = strategy
                    r['model'] = model
                    r['type'] = r_type
                    r['time'] = running_time
                    results = results.append(r)

                    results.to_csv(results_path)

                    predict['feed'] = r_id
                    predict['strategy'] = strategy
                    predict['model'] = model
                    predict['type'] = r_type
                    predictions = predictions.append(predict)

                    predictions.to_csv(forecast_path)

                    del (c, r2, mae, mape, r)
        # Compound Model
        # Un normalize
        # Re-seasonalize
        # Final forecast
        