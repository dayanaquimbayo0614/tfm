# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 17:55:19 2020

@author: dayan
"""


import os
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from plotnine import *
from plotnine.data import mpg
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
#Preferences
warnings.filterwarnings("ignore") 

sns.set_style('darkgrid')
plt.style.use('ggplot')

#Metrics
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
#Baseline
from sklearn.dummy import DummyRegressor
#Regresion
from sklearn import linear_model
#Random forest
from sklearn.ensemble import RandomForestRegressor
#Hyperparameter tuning
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#Tensorflow
import tensorflow as tf 




####################################

def retorno(data, n_dias):
    """
    Definición variable retorno, 0 en caso que el precio del indice en el día i sea 0,
    en caso contrario se calcula la variación con respecto al número de días indicados
    data: dataframe, conjunto de datos donde se quiere construir la variable
    n_dias: int, número de días en los que se quiere medir el retorno
    """
    data['lag_precio'+'_'+str(n_dias)] = data['precio'].shift(n_dias)
    name_var = 'retorno'+str(n_dias)
    data[name_var] = (data['precio']/data['lag_precio'+'_'+str(n_dias)]) -1
    data = data.replace([np.inf, -np.inf], 0)
    data = data.drop(['lag_precio'+'_'+str(n_dias)], axis = 1)
    return data

def target(data):
    """
    Definición variable retorno, 0 en caso que el precio del indice en el día i sea 0,
    en caso contrario se calcula la variación con respecto al número de días indicados
    data: dataframe, conjunto de datos donde se quiere construir la variable
    n_dias: int, número de días en los que se quiere medir el retorno
    """
    data['lag_precio_5diasf'] = data['precio'].shift(-5)
    data['lag_precio_6d'] = data['precio'].shift(6)
    data['lag_precio_1d'] = data['precio'].shift(1)
    data['target_5dias'] = (data['lag_precio_5diasf']/data['precio']-1)*100
    data = data.replace([np.inf, -np.inf], 0)
    data['predic_base'] = (data['lag_precio_6d']/data['lag_precio_1d']-1)*100
    data = data.drop(['lag_precio_5diasf'], axis = 1)
    return data

def estoch(data,x,n_sesiones, n_movil):
    #valor sugerido para n = 15, 20
    names = ['min', 'max', 'k_estock']
    names = [i+'_'+ x + '_' + str(n_sesiones) for i in names]
    data[names[0]] = data[x].rolling(window=n_sesiones).min()
    data[names[1]] = data[x].rolling(window=n_sesiones).max()
    data[names[2]] = (data[x]-data[names[0]])/(data[names[1]]-data[names[0]]) *100
    #data['media_movil'+'_'+str(n_movil)] = data[names[2]].rolling(n_movil).mean()
    data = data.drop([names[0], names[1]], axis = 1)
    return data

def creacion_variables(full_precios):

    full_precios = retorno(full_precios,5)
    full_precios = retorno(full_precios,15)
    full_precios = retorno(full_precios,30)
    full_precios['media_movil_5dias'] = full_precios['precio'].rolling(5).mean()
    full_precios['media_movil_15dias'] = full_precios['precio'].rolling(15).mean()
    full_precios['media_movil_30dias'] = full_precios['precio'].rolling(30).mean()
    full_precios['volatility_30'] = full_precios['retorno'].rolling(window=30).std() * np.sqrt(252)
    full_precios['volatility_60'] = full_precios['retorno'].rolling(window=60).std() * np.sqrt(252)
    full_precios['volatility_180'] = full_precios['retorno'].rolling(window=180).std() * np.sqrt(252)
    full_precios['volumen_1semana'] = full_precios['volumen'].rolling(min_periods=1, 
                                                              window=5).sum()
    full_precios['volumen_1mes'] = full_precios['volumen'].rolling(min_periods=1, 
                                                          window=20).sum()
    full_precios = target(full_precios)
    full_precios = estoch(full_precios,'precio',15, 3)
    full_precios = estoch(full_precios,'precio',30, 3)
    full_precios = estoch(full_precios,'precio',60, 3)
    full_precios = estoch(full_precios,'precio',90, 3)
    full_precios = estoch(full_precios,'retorno',5, 3)
    full_precios = estoch(full_precios,'retorno',15, 3)
    full_precios = estoch(full_precios,'retorno',20, 3)
    full_precios = estoch(full_precios,'retorno',60, 3)
    return full_precios

def variables_por_indice(data):
    for j,i in enumerate(data.indice.unique()):
        indice_precios = data[data.loc[:,'indice'] == i]
        if j == 0:
            data_indices = creacion_variables(indice_precios)
        else:
            #data_indices = []
            indice_precios_f = creacion_variables(indice_precios)
            data_indices = data_indices.append(indice_precios_f)
    return data_indices


def creacion_base(data):
    full_precios = data[data.date.dt.dayofweek < 5]
    full_precios_total = variables_por_indice(full_precios)
    full_precios_total = full_precios_total[full_precios_total.peso != 0]
    full_precios_total = full_precios_total[(full_precios_total.date >= '2015-01-01 00:00:00') & (full_precios_total.date <= '2019-12-24 00:00:00')]
    #data_ibex = pd.get_dummies(full_precios_total, columns = [ 'indice', 'sector'])
    #data_ibex = data_ibex.sort_values(by = ['date'])
    return full_precios_total

def creacion_base_ts(data):
    full_precios = data[data.date.dt.dayofweek < 5]
    full_precios_total = variables_por_indice(full_precios)
    full_precios_total = full_precios_total[full_precios_total.peso != 0]
    full_precios_total = full_precios_total[(full_precios_total.date >= '2015-01-01 00:00:00') & (full_precios_total.date <= '2019-12-24 00:00:00')]
    return full_precios_total

def data_visualizacion(data, size_train):
    full_precios = data[data.date.dt.dayofweek < 5]
    full_precios_total = variables_por_indice(full_precios)
    full_precios_total = full_precios_total[full_precios_total.peso != 0]
    full_precios_total = full_precios_total[(full_precios_total.date >= '2015-01-01 00:00:00') & (full_precios_total.date <= '2019-12-24 00:00:00')]
    full_precios_total = full_precios_total.sort_values(by = ['date'])
    n_train = int(full_precios_total.shape[0]*size_train)
    data_fin_train = full_precios_total.iloc[n_train, 0]
    TRAIN = full_precios_total[full_precios_total.date < data_fin_train] #Sin incluir fecha limite
    TEST = full_precios_total[full_precios_total.date >= data_fin_train]
    return TRAIN, TEST

def train_test(data):
    data_fin_train = '2019-03-01 00:00:00'
    TRAIN = data[data.date < data_fin_train] #Sin incluir fecha limite
    TEST = data[data.date >= data_fin_train]
    X_train = TRAIN.drop(['target_5dias', 'date', 'D1_ward', 'PAM', 'D2_complete', 'indice', 'sector'], axis = 1)
    y_train = TRAIN.loc[:,'target_5dias']
    X_test = TEST.drop(['target_5dias',  'date', 'D1_ward', 'PAM', 'D2_complete', 'indice', 'sector'], axis = 1)
    y_test = TEST.loc[:,'target_5dias']
    return X_train, y_train, X_test, y_test


def train_test_base(data):
    data_fin_train = '2019-03-01 00:00:00'
    TRAIN = data[data.date < data_fin_train] #Sin incluir fecha limite
    TEST = data[data.date >= data_fin_train]
    X_train = TRAIN.drop(['target_5dias','sector' ], axis = 1)
    y_train = TRAIN.loc[:,'target_5dias']
    X_test = TEST.drop(['target_5dias', 'sector'], axis = 1)
    y_test = TEST.loc[:,'target_5dias']
    return X_train, y_train, X_test, y_test


def normaliza(var):
    list_var = [list(var)]
    norma_list = preprocessing.normalize(list_var)
    norma_list_f = norma_list[0].T
    return norma_list_f


def volumen_precio(indice, n_dias):
    target_normal = normaliza(train_visualizacion[train_visualizacion.indice == indice].target_5)
    volumen_normal = normaliza(train_visualizacion[train_visualizacion.indice == indice].volumen)

    plt.figure(figsize=(20,5))
    plt.plot(train_visualizacion[train_visualizacion.indice == indice].date[:n_dias], volumen_normal[:n_dias], label='volumen')
    plt.plot(train_visualizacion[train_visualizacion.indice == indice].date[:n_dias], target_normal[:n_dias], label='target')
    legend = plt.legend(loc='upper center', shadow=True, fontsize='x-large')

    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('C0')
    plt.show()

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


def eval_model(X_train, X_test, y_train, y_test, model, name_model):
    _model = model
    _model.fit(X_train, y_train)
    y_predict = _model.predict(X_test)
    models_metrics[name_model] = model_metrics(y_test,y_predict)
    
    plt.figure(figsize=(20,5))
    plt.plot(y_test[-200:].reset_index(drop = True))
    plt.plot(y_predict[-200:])
    
    
full_precios = pd.read_excel('retornos_data.xlsx')
total_clust = pd.read_excel('total_clust.xlsx')
trends = pd.read_excel('data_trends_total.xlsx')


trends_data = trends.drop(['Fomento construcciones y contratas SA', 'Unnamed: 0'], axis = 1)
full_precios_clust = full_precios.merge(total_clust, on = 'indice', how = 'left')
full_precios_clust = full_precios_clust.merge(trends_data, on = 'date', how = 'left')
data_ibex = creacion_base(full_precios_clust)






