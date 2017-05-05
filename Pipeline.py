# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:57:10 2017

@author: Gonxo
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors as knn
from sklearn import neural_network
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.externals import joblib

import pandas as pd
import numpy as np
import os
import time

import mlfunctions as mlf
import featuresGenerator as ft

# Loading Data
# Obtaining working directory
wd = os.getcwd()
# Set up the destination  and secrects directory
# Set up the destination directory
dataDir = os.path.join(wd,'DATA_DIRECTORY') 

# User Apikey and feed ids are neccesary to get the feed's data.
# Writting and reading Apikeys are needed in case there is not a reading apikey bypass. 
# Direct to 'apiKeyDictionary.csv' location.
secretsDir = os.path.join(wd,'SECRETS_DIRECTORY')
apiDic = pd.read_csv(os.path.join(secretsDir,'apiKeyDictionary.csv'),sep=None, engine='python')
dataFile = 'raw_feeds'

# If you already have downloaded the data set to False.
fetch_data = False
fetch_weather_data = False

if fetch_data:
    mlf.getting_saving_data(dataDir,secretsDir, apiDic, dataFile)
    apiDic = pd.read_csv(os.path.join(secretsDir,'apiKeyDictionary.csv'),sep=None, engine='python')
    
if fetch_weather_data :
    runfile(os.path.join(wd,'obtaining_weather_data.py'), wdir=wd)

# Group by 15 min and 30 min

for grouper in ['15min', '30min']:
    
    window = 672
    h = 96
    if grouper == '30min':
        window = 672
        h = 48
        
    # Saving Results object
    results_path = os.path.join(dataDir,'RESULTS','%s.csv' %grouper)    
    columns = ['feed','type','strategy','model', 'measure']
    
    for i  in range(h):
        columns.append('t_%i' %(i+1))
    results = pd.DataFrame(columns= columns)
    predictions = pd.DataFrame(columns=columns)
    del predictions['measure']

    for index, row in apiDic.ix[:,['key','type','id']].iterrows():

        r_type = str(row['type'])
        r_id = str(row['id'])
        r_key = str(row['key'])
        print(r_id)
       
# Nas cleaning takes place in the next step.
# Solar or Solar + Grid with Nas cleaned.

        raw_feed = mlf.aggregation(r_key, r_type, r_id, os.path.join(dataDir, '%s.h5' %dataFile), apiDic)
        
        feed = raw_feed.groupby(pd.TimeGrouper(freq=grouper)).sum()
    
        # Deseasonalize data
        feed = mlf.deseasonalize(feed.ix[1:], r_id, plot=False)
        
        # Normalize data
        
        feed['normalized'] = ((feed.ix[:,-1] - feed.ix[:,-1].min()) / (feed.ix[:,-1].max() - feed.ix[:,-1].min()))+0.00000000001

        # Creating features, joining with weather data.

        features, response = ft.featureCreation(feed.ix[feed.ix[:,-1].first_valid_index():,-1],
                                                window, h, grouper,
                                                dataDir, apiDic, r_id)
        
        train = range(int(features.shape[0]*0.8))
        test = range(int(features.shape[0]*0.8), features.shape[0])
        
        # MIMO, DIR and Recursive
        
        for strategy in ['REC', 'DIR', 'MIMO']:

            if strategy == 'MIMO':
                
            # NN, KNN, RandForest. (SVR)
                for j in range(0,3):
                    if j == 2:
                        clf = neural_network.MLPRegressor(hidden_layer_sizes=(600,300,100),learning_rate='adaptive',alpha=0.01, max_iter=200, verbose=True ,batch_size=2000)
                        model = 'NN'
                    if j == 1:
                        clf = RandomForestRegressor(max_depth=30,random_state=0, verbose=True, n_jobs=-1)
                        model = 'RandFor'
                    if j== 0:
                        model = 'KNNR'
                        clf = knn.KNeighborsRegressor(n_neighbors=20 , n_jobs=-1)
                        
                    print ('Ready to run %s_%s_%s_%s' %(model,r_type,r_id, grouper))
                    start = time.time()
                    clf.fit(features[train],response[train])
                    print(time.time() - start)
                    
                    clf.score(features[test],response[test])
                    joblib.dump(clf, os.path.join(wd,'MODELS','%s_%s_%s_%s.pkl' %(model, r_type,r_id, grouper)))
                    
                    prediction = pd.DataFrame(clf.predict(features[test]), columns = ['t_%i' %(i+1) for i in range(h)])
                    
                    for measure in ['R2', 'MAE', 'MAPE']:
                        if measure == 'R2':
                            scores = r2_score(response[test], prediction, multioutput = 'raw_values')
                        if measure == 'MAE':
                            scores = mean_absolute_error(response[test], prediction, multioutput = 'raw_values')
                        if measure == 'MAPE':
                            scores = np.mean((np.abs(response[test] - prediction)/ response[test]), axis=0)
                    
                        c = [r_id, r_type, strategy, model, measure] 
                        c.extend(scores)

                        results = results.append(pd.DataFrame(c, index=columns).transpose())

                        results.to_csv(results_path)
                    
                    prediction['feed']= r_id
                    prediction['strategy'] = strategy
                    prediction['model'] = model
                    prediction['type'] = r_type
                    predictions = predictions.append(prediction)
                    
                    del (c, clf, prediction,scores)
                    
            if strategy == 'DIR':
    
                for j in range(0,3):
                    if j == 2:
                        clf = neural_network.MLPRegressor(hidden_layer_sizes=(600,300,100),learning_rate='adaptive', max_iter=200, verbose=True ,batch_size=400)
                        model = 'NN'
                        
                    if j == 1:
                        clf = RandomForestRegressor(max_depth=30,random_state=0, verbose=True, n_jobs=-1)
                        model = 'RandFor'
                        
                    if j== 0:
                        model = 'KNNR'
                        clf = knn.KNeighborsRegressor(n_neighbors=20 , n_jobs=-1)
                
                    if j==3:
                        train = range(int(features.shape[0]*0.7), int(features.shape[0]*0.8))
                        model = 'DIR_SVR'
                        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1],
                             'C': [ 10], 'epsilon':[0.1,0.15] }]
    #                    clf = GridSearchCV(SVR(cache_size=2200), param_grid= tuned_parameters,n_jobs=2, verbose = 5)
    #                        clf = SVR(C=10, epsilon=0.001, cache_size= 2500, gamma= 1, tol =0.0000001)
          
                    predict = pd.DataFrame()
                    r2 = ['R2']
                    mae = ['MAE']
                    mape = ['MAPE']
                    for i in range(96):
                        print ('Ready to run %s_%s_%s_t%d' %(model,r_type,r_id,(i+1)))
                        start = time.time()
                        clf.fit(features[train],response[train,i])
                        print(time.time() - start)
                        # print(clf.score(features[test],response[test]))
                        joblib.dump(clf, os.path.join(wd,'MODELS','%s_%s_%s_t%d.pkl' %( model, r_type,r_id,(i+1))))
    
 
                        prediction = pd.DataFrame(clf.predict(features[test]), columns= ['t_%i'%(i+1)])
    
                        predict = pd.concat([predict,prediction],axis=1)
    
                        for measure in ['R2', 'MAE', 'MAPE']:
                            if measure == 'R2':
                                r2.extend(r2_score(response[test,i], predict, multioutput = 'raw_values'))
                            if measure == 'MAE':
                                mae.extend(mean_absolute_error(response[test,i], predict, multioutput = 'raw_values'))
                            if measure == 'MAPE':
                                mape.extend([np.mean((np.abs(response[test,i] - predict)/ response[test,i]), axis=0)])
                            
                    c = ['measure']
                    c.extend(['t_%i' %(i+1) for i in range(h)])
                    r = pd.DataFrame([r2,mae,mape], columns =c)
                  
                    r['feed']= r_id
                    r['strategy'] = strategy
                    r['model'] = model
                    r['type'] = r_type
                    results = results.append(r)
     
                    results.to_csv(results_path)
    
                    predict['feed']= r_id
                    predict['strategy'] = strategy
                    predict['model'] = model
                    predict['type'] = r_type
                    predictions = predictions.append(predict)
                    
                    del (c, clf, r2, mae, mape, r)
                    
            if strategy == 'REC':
                
                for j in range(0,3):
                    if j == 2:
                        clf = neural_network.MLPRegressor(hidden_layer_sizes=(600,300,100),learning_rate='adaptive', max_iter=200, verbose=True ,batch_size=400)
                        model = 'NN'
                        
                    if j == 1:
                        clf = RandomForestRegressor(max_depth=30,random_state=0, verbose=True, n_jobs=-1)
                        model = 'RandFor'
                        
                    if j== 0:
                        model = 'KNNR'
                        clf = knn.KNeighborsRegressor(n_neighbors=20 , n_jobs=-1)
                        
                    print ('Ready to run %s_%s_%s_t%d' %(model,r_type,r_id,(1)))
                    start = time.time()
                    clf.fit(features[train],response[train,0])
                    print(time.time() - start)

                    joblib.dump(clf,os.path.join(wd,'MODELS','%s_%s_%s_t1.pkl' %( model, r_type,r_id)))
                    
                    feat = features
                    for i in range(h):
                        print('predicting %s' %(i+1))

                        feat[test[(i+1):-(h-i)],(h-i)] = clf.predict(feat[test[i:-(h+1-i)]]) 
                        
                    
                    predict = pd.DataFrame(np.flip(feat[test[h:],(feat.shape[1]-h):],1))
                    predict.columns = ['t_%i' %(i+1) for i in range(h)]
                    
                    for measure in ['R2', 'MAE', 'MAPE']:
                        if measure == 'R2':
                            scores = r2_score(response[test[h:]], predict, multioutput = 'raw_values')
                        if measure == 'MAE':
                            scores = mean_absolute_error(response[test[h:]], predict, multioutput = 'raw_values')
                        if measure == 'MAPE':
                            scores = np.mean((np.abs(response[test[h:]] - predict)/ response[test[h:]]), axis=0)
                    
                        c = [r_id, r_type, strategy, model, measure] 
                        c.extend(scores)

                        results = results.append(pd.DataFrame(c, index=columns).transpose())
                    
                        results.to_csv(results_path)
                    
                    predict['feed']= r_id
                    predict['strategy'] = strategy
                    predict['model'] = model
                    predict['type'] = r_type
                    predictions = predictions.append(predict[h+1,:])
                    
                    
        # Compound Model 
        
        # Un normalize
        
        # Re-seasonalize
        
        # Final forecast 
    
