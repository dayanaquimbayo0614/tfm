# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 16:24:38 2020

@author: dayan
"""
#Metrics
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

######################## FUNCIONES EVALUACIONES MODELO #######################

def normaliza(var):
    list_var = [list(var)]
    norma_list = preprocessing.normalize(list_var)
    norma_list_f = norma_list[0].T
    return norma_list_f

def model_metrics(y_test, predict):
    # Baseline errors, and display average baseline error
    baseline_errors = abs(predict - y_test)
    ape = np.where(y_test == 0, 0, 100*abs((y_test - predict)/y_test))
    #Mean avarage error
    print('Average baseline error: ', round(np.mean(baseline_errors), 4))
    # Error Cuadratico Medio
    print("Mean squared error: %.4f" % mean_squared_error(predict, y_test))
    # Raiz Error Cuadratico Medio
    print("Root mean squared error: %.4f" % np.sqrt(mean_squared_error(predict, y_test)))
    # R2
    print('R^2: %.5f' % r2_score(y_test, predict))
    #MAPE
    print('MAPE: %.5f' % np.mean(ape))
    print('MEDIANape: %.5f' % np.median(ape))
    accuracy = 100 -  mean_squared_error(predict, y_test)
    print('Accuracy:', round(accuracy, 2), '%.')
    return  mean_squared_error(predict, y_test)
    
def manual_split(dates, df, n_splits):
    
    splits = []
    for i in range(n_splits,0,-1):
        if i == n_splits:
            inicio_test = dates.max() - np.timedelta64(1, 'M')
            test_dates = dates[dates >= inicio_test] #Ultimos pred_horizon(int) datos
            inicio_train = inicio_test - relativedelta(years= 1) 
            train_dates = dates[(dates >= inicio_train) & (dates < inicio_test)]
            
        else:
            inicio_test = dates.max() - np.timedelta64(n_splits-(i-1), 'M') #mueve de mes en mes
            fin_test = inicio_test + np.timedelta64(1, 'M')
            test_dates = dates[(dates >= inicio_test) & (dates < fin_test)]
            inicio_train = inicio_test - relativedelta(years= 1)
            train_dates = dates[(dates >= inicio_train) & (dates < inicio_test)]
            
        splits.append((train_dates, test_dates))
        
    splits = splits[-1::-1]
        
    for i, split in enumerate(splits):
        dates_train = split[0]
        dates_test = split[1]
        train_index=df.loc[df.date.isin(dates_train)].index.values
        test_index=df.loc[df.date.isin(dates_test)].index.values
        print('Train Min{} Max{}'.format(train_index.min(),train_index.max()))
        print('Test Min{} Max{}'.format(test_index.min(),test_index.max()))
        yield (np.array(train_index), np.array(test_index))


def eval_model(X_train, X_test, y_train, y_test, model, name_model, dic_metrics):
    _model = model
    _model.fit(X_train, y_train)
    y_predict = _model.predict(X_test)
    dic_metrics[name_model] = model_metrics(y_test,y_predict)
    
    plt.figure(figsize=(20,5))
    plt.plot(y_test[-200:].reset_index(drop = True))
    plt.plot(y_predict[-200:])
    