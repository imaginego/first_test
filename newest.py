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
addr_fea_ngram = AddressNgramFeature(10)

quant_fea = PriceQuantileFeature()

gbm_quant_fea = GbmQuantPrice(['days','latitude','longitude'],'gbm_quant_lat_long')
gbm_id_fea = GbmQuantPrice(['building_id','manager_id'],'gbm_quant_id',n_estimators=200,max_depth=5)
mis_fea = Miscellous()

#print '-------------new addr feature--------------'
#xgb = XGB_model(train_df,test_df,feature_list = [txt_fea,addr_fea_ngram,cat_lit_fea,gbm_quant_fea,mis_fea],target_train= 'interest_level')
#xgb.cv_study(nImportance=10,n_estimators=5000)

#print '-------------old addr feature---------------'
#xgb = XGB_model(train_df,test_df,feature_list = [txt_fea,addr_fea,cat_cv_fea,gbm_quant_fea,mis_fea],target_train= 'interest_level')
#xgb.cv_study(nImportance=10,n_estimators=5000)

xgb_param = {'n_estimators':2800,
             'objective':'multi:softprob',
             'learning_rate':0.01,
             'max_depth':6,
             'min_child_weight':1,
             'subsample':.7,
             'colsample_bytree':.7,
             'colsample_bylevel':.5,
             'gamma':0.005,
             'scale_pos_weight':1,
             'base_score':.5
             }

stack1 = Stacking_model(train_df,test_df,feature_list=[txt_fea,addr_fea,gbm_quant_fea,mis_fea],
                        target_train= 'interest_level',classifier1 = 'random forest',param1={'n_estimators':300},
                        classifier2='xgboost',param2=xgb_param)
preds = stack1.fit_predict_proba(3)
write_output(preds,test_df,prefix='stack_rf_xgb_nocv')

stack2 = Stacking_model(train_df,test_df,feature_list=[txt_fea,addr_fea,gbm_quant_fea,mis_fea],
                        target_train= 'interest_level',classifier1 = 'extra trees',param1={'n_estimators':300},
                        classifier2='xgboost',param2=xgb_param)
preds = stack2.fit_predict_proba(3)
write_output(preds,test_df,prefix='stack_extree_xgb_nocv')


stack3 = Stacking_model(train_df,test_df,feature_list=[txt_fea,addr_fea,gbm_quant_fea,mis_fea],
                        target_train= 'interest_level',classifier1 = 'gradient boosting',
                        param1={'n_estimators':300,'max_depth':5,'learning_rate':0.05},
                        classifier2='xgboost',param2=xgb_param)
preds = stack3.fit_predict_proba(3)
write_output(preds,test_df,prefix='stack_gbm_xgb_nocv')

