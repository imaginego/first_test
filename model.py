from feature_extraction import *
import numpy as np
import pandas as pd
from sklearn import model_selection 
import xgboost as xgb
from xgboost import XGBClassifier,XGBRegressor
from sklearn import preprocessing, ensemble
from sklearn.metrics import log_loss


class BaseModel(object):
    def __init__(self,data_train,data_test,feature_list,target_train='interest_level',sample_weight = None,	
		  ):
        self.sample_weight = sample_weight
        self.feature_names = ["bathrooms", "bedrooms", "latitude", "longitude", "price",'listing_id'] 
        self.feature_list = feature_list
        self.classifier = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        
        self.get_feature(data_train,target_train,data_test)
    
    def set_classifier(self):
        if self.classifier is not None:
            del self.classifier
		
    
    def get_feature(self,data_train,target_train,data_test):
        self.X_train = data_train.copy()
        
        self.X_test = data_test.copy()
		#self.feature_names =  
        for ff in self.feature_list:
            print ff.get_info()
            self.X_train,self.X_test,tmp= ff.transform(self.X_train,self.X_test)
            print len(tmp)
            self.feature_names += tmp
            
        #import pdb;pdb.set_trace()
        self.y_train = self.X_train[target_train].as_matrix().ravel()
        self.X_train = self.X_train[self.feature_names].as_matrix()
        self.X_test = self.X_test[self.feature_names].as_matrix()	
        
        
    def cv_study(self,nImportance=0, random_state=2016):
        if self.classifier is None:
            raise NameError("Classifier does not exist")
        cv_scores = [] 
        kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
        for dev_index, val_index in kf.split(self.X_train,self.y_train):
            dev_X, val_X = self.X_train[dev_index,:], self.X_train[val_index,:]
            dev_y, val_y = self.y_train[dev_index], self.y_train[val_index]
            self.set_classifier()
            self.classifier.fit(dev_X,dev_y)

            if nImportance:
                inx = np.argsort((-1)*self.classifier.feature_importances_)
                print("The most {} important features are".format(nImportance))
                for ii in inx:
                    print '{}, importance={}'.format(self.feature_names[ii],self.classifier.feature_importances_[ii])
            preds = self.classifier.predict_proba(val_X)
            cv_scores.append(log_loss(val_y, preds))
            print cv_scores
        print 'mean score={}'.format(np.mean(cv_scores))
        print 'std = {}'.format(np.std(cv_scores))

    def fit_predict(self):
		if self.classifier is None:
			raise NameError("Classifier does not exist")

		self.classifier.fit(self.X_train,self.y_train)		
		return self.classifier.predict(self.X_test)
    
    def fit_predict_proba(self):
        if self.classifier is None:
            raise NameError("Classifier does not exist")
        self.classifier.fit(self.X_train,self.y_train)		
        return self.classifier.predict_proba(self.X_test)


class XGB_model(BaseModel):
    def __init__(self,data_train,data_test,feature_list,target_train):
        super(XGB_model,self).__init__(data_train,data_test,feature_list,target_train)
        self.best_rounds = 3000
    
    def set_classifier(self,n_estimators,
                            objective='multi:softprob',
                            learning_rate=0.01,
                            max_depth=6,
                            min_child_weight=1,
                            subsample=.7,
                            colsample_bytree=.7,
                            colsample_bylevel=.5,
                            gamma=0.005,
                            scale_pos_weight=1,
                            base_score=.5,
                            #reg_lambda=0,
                            #reg_alpha=0,
                            #missing=0,
                            seed=0):
        super(XGB_model,self).set_classifier()
        self.classifier = XGBClassifier(n_estimators=n_estimators,
                            objective=objective,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            min_child_weight=min_child_weight,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            colsample_bylevel=colsample_bylevel,
                            gamma=gamma,
                            scale_pos_weight=scale_pos_weight,
                            base_score=base_score,
                            #reg_lambda=0,
                            #reg_alpha=0,
                            #missing=0,
                            seed=seed)
        
    def cv_study(self,nImportance=0,n_estimators = 5000,random_state=2016,verbose=False,early_stopping_rounds=50):
        cv_scores = [] 
        kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        for dev_index, val_index in kf.split(self.X_train,self.y_train):
            dev_X, val_X = self.X_train[dev_index,:], self.X_train[val_index,:]
            dev_y, val_y = self.y_train[dev_index], self.y_train[val_index]
            if self.sample_weight is not None:
                dev_w, val_w = self.sample_weight[dev_index], self.sample_weight[val_index]
            else:
                dev_w,val_w = None,None
                
            #import pdb;pdb.set_trace()
            self.set_classifier(n_estimators)
            self.classifier.fit(dev_X, dev_y,sample_weight= dev_w,eval_set=[(dev_X, dev_y), (val_X, val_y)],verbose=verbose,eval_metric='mlogloss', early_stopping_rounds = early_stopping_rounds)
            
            tmp = self.classifier.best_iteration
            self.best_rounds = tmp if tmp>self.best_rounds else self.best_rounds
            if nImportance:
                inx = np.argsort((-1)*self.classifier.feature_importances_)
                print("The most {} important features are".format(nImportance))
                for ii in inx[:nImportance]:
                    print '{}, importance={}'.format(self.feature_names[ii],self.classifier.feature_importances_[ii])
        		
            preds = self.classifier.predict_proba(val_X)
            cv_scores.append(log_loss(val_y, preds))
            print cv_scores
            
            print('best iterations:{}, best_score={}, last_score={}'.format(self.classifier.best_iteration,self.classifier.best_score,log_loss(val_y, preds)))
        print cv_scores
        print 'mean score={}'.format(np.mean(cv_scores))
        print 'std = {}'.format(np.std(cv_scores))

	def fit_predict(self):
		if self.classifier is None:
			raise NameError("Classifier does not exist")
		self.classifier.set_params({'n_estimators':self.best_rounds})
		self.classifier.fit(self.X_train,self.y_train,sample_weight=self.sample_weight,verbose=False)		
		return self.classifier.predict(self.X_test)

	def fit_predict_proba(self):
		if self.classifier is None:
			raise NameError("Classifier does not exist")
		self.classifier.set_params({'n_estimators':self.best_rounds})
		self.classifier.fit(self.X_train,self.y_train,sample_weight=self.sample_weight,verbose=False)		
		return self.classifier.predict_proba(self.X_test)
