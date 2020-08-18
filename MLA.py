from keras.models import load_model
import numpy as np
import pandas as pd
from django.contrib import messages
from pandas import DataFrame
from pandas import concat
import pickle
import os
import joblib





csv_filepath = './media/prepared.csv'
model_filepath = './media/'
scaler_filepath = './media/scaler.pickle'


def series_to_supervised(data, n_in=50, n_out=1, dropnan=True):
	if type(data) is list:
		n_vars = 1
	else:
		n_vars = data.shape[1]
	# print("特征个数n_vars为:", n_vars)
	df = DataFrame(data)
	cols, names = list(), list()
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		for j in range(n_vars):
			names.append('var%d(t-%d)' % (j + 1, i))
	df.drop(df.columns[[1, 2, 3, 4, 5, 6, 7]], axis=1, inplace=True)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		for j in range(1):
			names.append('var%d(t+%d)' % (j + 1, i))
	agg = concat(cols, axis=1)
	agg.columns = names
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def data_pre(df):
	# data = pd.read_csv("./datatest.csv")
	# data = data.drop('t', axis=1)
	#data = dataset.drop('time_idx', axis=1)[-100:] # take the last 100 data for testing
	data= df
	values = data.values
	values = values.astype('float32')
	reframed = series_to_supervised(values, 5, 1)
	test = reframed.values
	test_X, test_y = test[:, :-1], test[:, -1]
	test_y = test_y.reshape(-1, 1)
	return test_X,test_y


def predict_LSTM(df):   #长短期记忆神经网络
	test_X,test_y = data_pre(df)
			#use the training scalar to scale the test dataset
	with open(scaler_filepath, 'rb') as fr:
		new_scaler = pickle.load(fr)
	test_X = new_scaler.transform(test_X)
	test_y = new_scaler.transform(test_y)
	#	reshape the data form to feed into LSTM
	test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
	new_model = load_model(os.path.join(model_filepath,'LSTM.h5'))  #load model
	pre_test = new_model.predict(test_X)  #predict
	pre_test = pre_test.flatten()
	test_y = test_y.flatten()
	return pre_test, test_y

def predict_LGBM(df):
	test_X, test_y = data_pre(df)
	with open(scaler_filepath, 'rb') as fr:
		new_scaler = pickle.load(fr)
	test_X = new_scaler.transform(test_X)
	test_y = new_scaler.transform(test_y)
	new_model=joblib.load(os.path.join(model_filepath,'lightgbm.model'))
	pre_test = new_model.predict(test_X)  #predict
	pre_test = pre_test.flatten()
	test_y = test_y.flatten()
	return pre_test, test_y

def predict_RandomForest(df):
	test_X, test_y = data_pre(df)
	with open(scaler_filepath, 'rb') as fr:
		new_scaler = pickle.load(fr)
	test_X = new_scaler.transform(test_X)
	test_y = new_scaler.transform(test_y)
	new_model=joblib.load(os.path.join(model_filepath,'randomforest.model'))
	pre_test = new_model.predict(test_X)  #predict
	pre_test = pre_test.flatten()
	test_y = test_y.flatten()
	return pre_test, test_y

def predict_Adaboost(df):
	test_X, test_y = data_pre(df)
	with open(scaler_filepath, 'rb') as fr:
		new_scaler = pickle.load(fr)
	test_X = new_scaler.transform(test_X)
	test_y = new_scaler.transform(test_y)
	new_model=joblib.load(os.path.join(model_filepath,'Adaboost.model'))
	pre_test = new_model.predict(test_X)  #predict
	pre_test = pre_test.flatten()
	test_y = test_y.flatten()
	return pre_test, test_y








