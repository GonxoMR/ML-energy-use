# -*- coding: utf-8 -*-
"""
Created on Tue May 30 10:39:33 2017

@author: Gonxo
"""
def findStrategy(horizon, size):
    if horizon == 1:
        strategy = 'Recursive'
    elif size == 1:
        strategy = 'Direct'
    elif size < horizon:
        strategy = 'DIRMO'
    else:
        strategy = 'MIMO'
    return strategy

def train_forecast_strategies(features, response, model, horizon=1, size=1):
    """
    This function computes finds the best parameters for the models and strategy
    selected. With this function the strategies Recursive, Direct, MIMO and Dirmo
    can be explored by tunning the parameters 'horizon' and'size':

    Recursive: 'horizon = 1', 'size = 1'.
    Direct: 'horizon > 1', 'size = 1'.
    DIRMO: 'horizon > 1', '1 < size < horizon'.
    MIMO: 'horizon > 1', 'size = horizon'.

    It returns an array of size horizon/size containing the trainded models corresponding
    to the strategy.

    Input:
        features: matrix containing the desired features for the given dataset.
        response: array or matrix of size h x n. Where h is the desired horizon and
                  n the numbero ob observations.
        model: is a scikit-learn model to be train.
        horizon: int referencing how many steps ahead you want to forecast.
        size: int how many steps ahead each model forecast.

    Output:
        models: list containing as the first element the size of the forecast
                produced by the models and then the different models.
    """
    import time
    import os
    import sys
    import warnings
    from sklearn.externals import joblib

    if horizon > response.shape[1]:
        sys.exit('The set horizon is larger than the available steps ahead in response.')
    if size > horizon:
        warnings.warn('Warning: the size of the chunks is bigger than the horizon. MIMO assumed')
        size = horizon

    start = time.time()

    models = []

    strategy = findStrategy(horizon, size)

    for i in range(int(horizon/size)):
        print('Ready to run %s_t%d' %(strategy, ((i+1)*size)))

        models.append(model.fit(features, response[:, (i*size):((i+1)*size)]))

        joblib.dump(model, os.path.join(os.getcwd(), '%s_t%d.pkl' %(strategy,
                                                                    (i*size)+1)))

    running_time = time.time() - start
    print(running_time/60)

    return models

def forecast(feature, models=None, horizon=48, size=48, strategy='MIMO'):
    """
    This performs and returns the forecast for a given period of time. The inputs
    are the past values of the series and the forecast weather data for the future
    period merged in the feeds and the trained models.

    The output is a dataframe with the forecast produced from moment t+1 to t+size.
    """
    import pandas as pd
    import sys
    import os
    from sklearn.externals import joblib

    if models is None:
        if os.path.isfile(os.path.join(os.getcwd(), '%s_t%d.pkl' %(strategy, 1))):
            model = []
            for i in range(int(horizon/size)):
                if os.path.isfile(os.path.join(os.getcwd(), '%s_t%d.pkl' %(strategy, (i*size)+1))):
                    models.append(joblib.load(os.path.join(os.getcwd(),
                                                           '%s_t%d.pkl' %(strategy,
                                                                          (i*size)+1))))
        else:
            sys.exit('There are no models trained for the %s strategy.' %strategy)

    predict = pd.DataFrame()

    for i, model in enumerate(models):
        prediction = pd.DataFrame(model.predict(feature),
                                  columns=['t_%i'%(j+1) for j in range((i*size),
                                                                       ((i+1)*size))])
        predict = pd.concat([predict, prediction], axis=1)

    return predict
