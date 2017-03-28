import keras

import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import hashlib
import re
from collections import defaultdict
from sklearn import preprocessing

from keras.layers import Merge
from keras.models import Sequential
from keras.layers import Dense, Activation,Lambda,Dropout
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.layers import Input
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from keras.regularizers import l2

def runNN(tr_text_fea, tr_quant_fea, train_y, te_text_fea=None, te_quant_fea=None,test_y=None):

	left_branch = Sequential()
	compress_layer = Dense(120, input_dim=tr_text_fea.shape[1],activation='linear',bias=False)
	left_branch.add(compress_layer)

	right_branch = Sequential()
	right_branch.add(Lambda(lambda x: x, input_shape=(tr_quant_fea.shape[1],)))

	merged = Merge([left_branch, right_branch], mode='concat')

	final_model = Sequential()
	final_model.add(merged)
	#final_model.add(Dense(32,activation='tanh'))
	#final_model.add(Dense(64,activation='tanh'))#,W_regularizer=l2(0.01)))
	final_model.add(Dense(64,activation='tanh'))#,W_regularizer=l2(0.01)))
	final_model.add(Dense(64,activation='tanh'))#,W_regularizer=l2(0.01)))
	#final_model.add(Dense(64,activation='tanh'))#,W_regularizer=l2(0.01)))
	final_model.add(Dense(64,activation='tanh'))#,W_regularizer=l2(0.01)))
	final_model.add(Dense(3, activation='softmax'))
	#final_model.summary()

	optimizer = Adam(lr=1e-4) # Using Adam instead of SGD to speed up training
	final_model.compile(loss='categorical_crossentropy', optimizer=optimizer)

	lr_reducer      = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), 
                                    cooldown=0, patience=5, min_lr=0.1e-7)
	early_stopper   = EarlyStopping(monitor='val_loss', min_delta=0.00005, patience=30)
	#model_checkpoint= ModelCheckpoint(r"weights/DenseNet-Fast-40-12-first.h5", monitor="val_acc", save_best_only=True,
    #                              save_weights_only=True,mode='auto')

	callbacks=[lr_reducer,early_stopper]
	score = 0

	if te_text_fea is None:
		final_model.fit([tr_text_fea, tr_quant_fea], train_y,nb_epoch=75,verbose=0)  # we pass one data array per model input
	else:
		final_model.fit([tr_text_fea, tr_quant_fea], train_y,validation_data=([te_text_fea,te_quant_fea],test_y),nb_epoch=250,verbose=0,callbacks=callbacks)  # we pass one data array per model input

		score = final_model.evaluate([te_text_fea,te_quant_fea],test_y)
	return score,compress_layer

def get_fea():
	data_path = "../input/"
	train_file = data_path + "train.json"
	test_file = data_path + "test.json"
	train_df = pd.read_json(train_file)
	test_df = pd.read_json(test_file)
	original_col = train_df.columns

	#colnames = ['features','latitude','longitude','price','bathrooms','bedrooms','interest_level']

	df = train_df.copy()

	df['features'] = df['features'].map(lambda x:[tt.lower() for tt in x])
	#df['description'] = df['description'].map(lambda x:x.lower())
    
	all_fea = defaultdict(int)
	for _,row in df.iterrows():
		for xx in row['features']:
			all_fea[xx] += 1

	
	fea_list = [k for (k,v) in all_fea.iteritems() if v>=3]

	fea_map = {fea_list[ii]:ii for ii in range(len(fea_list))}

	df = df.reset_index(drop=True)
	text_fea = np.zeros((len(df),len(fea_map)))
	for inx,row in df.iterrows():
		for xx in row['features']:
			if xx in fea_map:
				text_fea[inx][fea_map[xx]] = 1


	df["created"] = pd.to_datetime(df["created"])
	df["created_year"] = df["created"].dt.year
	
	df["created_month"] = df["created"].dt.month
	
	df["created_day"] = df["created"].dt.day
	
	df["created_hour"] = df["created"].dt.hour
	df["num_photos"] = df["photos"].apply(len)

	
	quant_fea = df[['latitude','longitude','price','bathrooms','bedrooms','created_month','created_day','created_hour','num_photos']].as_matrix()

	quant_fea = preprocessing.normalize(quant_fea)

	target_num_map = {'high':0, 'medium':1, 'low':2}
	y = np.array(df['interest_level'].apply(lambda x: target_num_map[x]))
	y = np_utils.to_categorical(y, nb_classes=3)

	return text_fea,quant_fea,y,fea_map


def test_compress(fea,fea_map):
	#fea = ['laundary in unit','laundry in unit','in-unit washer/dryer']

	text_fea,quant_fea,y,fea_map = get_fea()
	_,compress = runNN(text_fea,quant_fea,y)

	wei = compress.get_weights()
	import pdb;pdb.set_trace()
	ff = np.zeros((len(fea),len(fea_map)))
	for ii in range(len(fea)):
		if fea[ii] in fea_map:
			ff[ii,fea_map[fea[ii]]]=1
		else:
			print fea[ii]

	tmp = np.dot(ff,wei[0])
	print tmp




if __name__=='__main__':
	text_fea,quant_fea,y,fea_map = get_fea()
	test_compress(['washer/dryer in unit','laundry in unit','in-unit washer/dryer'],fea_map)

	
	cv_scores = []
	kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2016)
	for dev_index, val_index in kf.split(range(text_fea.shape[0])):
		dev_text_X, val_text_X = text_fea[dev_index,:], text_fea[val_index,:]
		dev_quant_X,val_quant_X = quant_fea[dev_index,:],quant_fea[val_index,:]

		dev_y, val_y = y[dev_index], y[val_index]
		cv_score = runNN(dev_text_X, dev_quant_X,dev_y, val_text_X, val_quant_X,val_y)
		cv_scores.append(cv_score)
		print(cv_scores)
		#break
	print 'mean score={}'.format(np.mean(cv_scores))
