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

cat_cv_fea = Categorical_cv(nfold = 5)

cat_lit_fea = CategoricalLit(nfold=5)

txt_fea = TextFeature()

addr_fea = AddressFeature()

quant_fea = PriceQuantileFeature()

gbm_quant_fea = GbmQuantPrice(['latitude','longitude'],'gbm_quant_lat_long')
mis_fea = Miscellous()

xgb = XGB_model(train_df,test_df,feature_list = [txt_fea,addr_fea,cat_lit_fea,gbm_quant_fea,mis_fea],target_train= 'interest_level')
xgb.cv_study(nImportance=10,n_estimators=5000)



