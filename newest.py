import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
from xgboost import XGBClassifier,XGBRegressor
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV,StratifiedKFold


from feature_extraction import *
from model import *
from utils import *

train_df,test_df = read_data()

cat_cv_fea = Categorical_cv(nfold = 5,bEncoding=True)

cat_lit_fea = CategoricalLit(nfold=5,bEncoding=True)

txt_fea = TextFeature()

addr_fea = AddressFeature()
addr_fea_ngram = AddressNgramFeature(10)

quant_fea = PriceQuantileFeature()

gbm_quant_fea = GbmQuantPrice(['latitude','longitude'],'gbm_quant_lat_long')
mis_fea = Miscellous()

print '-------------new addr feature--------------'
xgb = XGB_model(train_df,test_df,feature_list = [txt_fea,addr_fea_ngram,cat_lit_fea,gbm_quant_fea,mis_fea],target_train= 'interest_level')
xgb.cv_study(nImportance=10,n_estimators=5000)

print '-------------old addr feature---------------'
xgb = XGB_model(train_df,test_df,feature_list = [txt_fea,addr_fea,cat_lit_fea,gbm_quant_fea,mis_fea],target_train= 'interest_level')
xgb.cv_study(nImportance=10,n_estimators=5000)




