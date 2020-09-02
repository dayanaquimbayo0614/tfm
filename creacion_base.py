# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 17:55:19 2020

@author: dayan
"""


import pandas as pd
import numpy as np
import warnings
from pandas import concat
import pandas as DataFrame
from dateutil.relativedelta import relativedelta

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
#Preferences
warnings.filterwarnings("ignore") 






########################### FUNCIONES CREACIÓN BASES ##########################

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

def creacion_variables(data):

    data = retorno(data,5)
    data = retorno(data,15)
    data = retorno(data,30)
    data['media_movil_5dias'] = data['precio'].rolling(5).mean()
    data['media_movil_15dias'] = data['precio'].rolling(15).mean()
    data['media_movil_30dias'] = data['precio'].rolling(30).mean()
    data['volatility_30'] = data['retorno'].rolling(window=30).std() * np.sqrt(252)
    data['volatility_60'] = data['retorno'].rolling(window=60).std() * np.sqrt(252)
    data['volatility_180'] = data['retorno'].rolling(window=180).std() * np.sqrt(252)
    data['volumen_1semana'] = data['volumen'].rolling(min_periods=1, window=5).sum()
    data['volumen_1mes'] = data['volumen'].rolling(min_periods=1,window=20).sum()
    data = target(data)
    data = estoch(data,'precio',15, 3)
    data = estoch(data,'precio',30, 3)
    data = estoch(data,'precio',60, 3)
    data = estoch(data,'precio',90, 3)
    data = estoch(data,'retorno',5, 3)
    data = estoch(data,'retorno',15, 3)
    data = estoch(data,'retorno',20, 3)
    data = estoch(data,'retorno',60, 3)
    data = estoch(data,'predic_base',20, 3)
    data = estoch(data,'predic_base',40, 3)
    return data

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

def data_visualizacion(data):
#Se crea por aparte para hacer la creación de variables auxiliares y segemtar en train y test, pero teniendo la variable target en el conjunto de train. 
    full_precios = data[data.date.dt.dayofweek < 5]
    full_precios_total = variables_por_indice(full_precios)
    full_precios_total = full_precios_total[full_precios_total.peso != 0]
    full_precios_total = full_precios_total[(full_precios_total.date >= '2015-01-01 00:00:00') & (full_precios_total.date <= '2019-12-24 00:00:00')]
    full_precios_total = full_precios_total.sort_values(by = ['date'])
    data_fin_train = '2019-03-01 00:00:00'
    TRAIN = full_precios_total[full_precios_total.date < data_fin_train] #Sin incluir fecha limite
    TEST = full_precios_total[full_precios_total.date >= data_fin_train]
    return TRAIN, TEST


def creacion_base(data):
    full_precios = data[data.date.dt.dayofweek < 5]
    full_precios_total = variables_por_indice(full_precios)
    full_precios_total = full_precios_total[full_precios_total.peso != 0]
    full_precios_total = full_precios_total[(full_precios_total.date >= '2015-01-01 00:00:00') & (full_precios_total.date <= '2019-12-24 00:00:00')]
    return full_precios_total

def train_test(data):
    data_fin_train = '2019-03-01 00:00:00'
    TRAIN = data[data.date < data_fin_train] #Sin incluir fecha limite
    TEST = data[data.date >= data_fin_train]
    X_train = TRAIN.drop(['target_5dias', 'date', 'indice', 'sector'], axis = 1)
    y_train = TRAIN.loc[:,'target_5dias']
    X_test = TEST.drop(['target_5dias',  'date', 'indice', 'sector'], axis = 1)
    y_test = TEST.loc[:,'target_5dias']
    return X_train, y_train, X_test, y_test

def fechas_train_test(data):
    data_fin_train = '2019-03-01 00:00:00'
    train_total = data[data.date < data_fin_train]
    test_total = data[data.date >= data_fin_train]
    return train_total, test_total


   # convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Transformación base de datos al formato necesario para las redes LSTM.
	Variable target como principal, rezagos de la misma como auxiliares. 
	data: dataframe, dataframe con variable target (o varias variables).
	n_in: int, número de rezagos hacia atras que se quieren para el dataframe.
	n_out: int, número de rezagos hacia adelante que se quiren para el dataframe.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
		# put it all together
		agg = concat(cols, axis=1)
		agg.columns = names
		# drop rows with NaN values
		if dropnan:
			agg.dropna(inplace=True)
	return agg  

def manual_split(dates, df, n_splits):
 #Esta función realiza la segmentación adecuada para el gridsearch del modelo
 # 
    
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
        
        
#def transformacion_data(data):
#  values = data.values
#  encoder = LabelEncoder()
#  values[:,4] = encoder.fit_transform(values[:,4])
#  values = values.astype('float32')
#  scaler = MinMaxScaler(feature_range=(0, 1))
#  scaled = scaler.fit_transform(values)
#  reframed = series_to_supervised(scaled, 1, 1)
#  return reframed
    

#def tranformacion_por_indice(data):
#    for j,i in enumerate(data.indice.unique()):
#        indice_precios = data[data.loc[:,'indice'] == i]
#        if j == 0:
#            data_indices = transformacion_data(indice_precios)
#        else:
#            indice_precios_f = transformacion_data(indice_precios)
#            data_indices = data_indices.append(indice_precios_f)
#    data_indices.sort_values(by = 'var66(t)')
#    data_indices.drop(data_indices.columns[list(range(67,132))], axis=1, inplace=True)
#    return data_indices


################# CARGA DATOS & CREACIÓN TABLA INICIAL #######################
    
full_precios = pd.read_excel('retornos_data.xlsx')
total_clust = pd.read_excel('total_clust.xlsx')
trends = pd.read_excel('data_trends_total.xlsx')


full_precios_clust = full_precios.merge(total_clust, on = 'indice', how = 'left')
full_precios_clust_trend = full_precios_clust.merge(trends, on = 'date', how = 'left')
data_ibex = creacion_base(full_precios_clust_trend)

###################### SEGMENTACIÓN TRAIN - TEST ##############################


#Segmentación para entrenar los modelos: Sin variables target, fecha, sector, indice, predict_base en X.
X_train, y_train, X_test, y_test = train_test(data_ibex)

train_total, test_total = fechas_train_test(data_ibex)
###################### MODELOS RANDOM FOREST #####################################





######################### MODELOS POR CLUSTER #################################
full_precios_clust1 = data_ibex[data_ibex.D1_ward == 1]
full_precios_clust2 = data_ibex[data_ibex.D1_ward == 2]



####################### TRANSFORMACIÓN DATOS LSTM ############################

#Transformación a una serie secuencial
#data_total_transform = tranformacion_por_indice(data_ibex)


#Segmentación TRAIN - TEST de la serie secuencial
#(se hace por núm registros xq no hay fecha)
#values = data_total_transform.values
#n_train_days = 17500
#train = values[:n_train_days, :]
#test = values[n_train_days:, :]
# split into input and outputs
#train_X, train_y = train[:, :-1], train[:, -1]
#test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
#train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
#test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)



    
    
    
    
    
    
    
    
    
    