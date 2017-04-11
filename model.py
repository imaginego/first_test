from feature_extraction import *
import numpy as np
import pandas as pd
from sklearn import model_selection 
import xgboost as xgb
from xgboost import XGBClassifier,XGBRegressor
from sklearn import preprocessing, ensemble
from sklearn.metrics import log_loss,accuracy_score
import sklearn
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn import svm
preprocessing.normalize

class BaseModel(object):
    def __init__(self,data_train,data_test,feature_list,target_train='interest_level',sample_weight = None):
        self.sample_weight = sample_weight
        self.feature_names = ["bathrooms", "bedrooms", "latitude", "longitude", "price",'listing_id'] 
        self.feature_list = feature_list
        self.classifier = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        if len(feature_list)>0:
            self.get_feature(data_train,target_train,data_test)
    
    def reset_classifier(self):
        if self.classifier is not None:
            del self.classifier
		
    def set_classifier(self,**params):
        self.reset_classifier()
		
    def gridsearch_cv(self,param_list):
        pass
    
    def get_feature(self,data_train,target_train,data_test):
        self.X_train = data_train.copy()        
        self.X_test = data_test.copy()
        data_test['interest_level'] = np.nan
        anchor = data_test[['listing_id','interest_level']].copy() #used to keep X_test in same order of test_df
		  
        for ff in self.feature_list:
            print ff.get_info()
            self.X_train,self.X_test,tmp= ff.transform(self.X_train,self.X_test)
            #print len(tmp)
            if len(tmp)<50:
                print tmp
            self.feature_names += tmp
            
        self.y_train = self.X_train[target_train].as_matrix().ravel()
        self.X_train = self.X_train[self.feature_names].as_matrix()
        self.X_train,self.y_train = sklearn.utils.shuffle(self.X_train,self.y_train,random_state=200)
        self.X_test = pd.merge(anchor,self.X_test,on='listing_id',how='left')
        self.X_test = self.X_test[self.feature_names].as_matrix()	
    
    def set_data(self,X_train,X_test,y_train):
        self.X_test = X_test
        self.X_train = X_train
        self.y_train = y_train
        
    def cv_study(self,nImportance=0, random_state=2016,**params):
        #if self.classifier is None:
        #    raise NameError("Classifier does not exist")
        cv_scores = [] 
        acc_scores = []
        kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
        for dev_index, val_index in kf.split(self.X_train,self.y_train):
            dev_X, val_X = self.X_train[dev_index,:], self.X_train[val_index,:]
            dev_y, val_y = self.y_train[dev_index], self.y_train[val_index]
            self.set_classifier(**params)
            self.classifier.fit(dev_X,dev_y)

            if nImportance:
                inx = np.argsort((-1)*self.classifier.feature_importances_)
                print("The most {} important features are".format(nImportance))
                for ii in inx[:nImportance]:
                    print '{}, importance={}'.format(self.feature_names[ii],self.classifier.feature_importances_[ii])
            if hasattr(self.classifier,'predict_proba'):
                preds = self.classifier.predict_proba(val_X)
                cv_scores.append(log_loss(val_y, preds))
                print cv_scores
            cls_preds = self.classifier.predict(val_X)
            acc = accuracy_score(val_y,cls_preds)
            acc_scores.append(acc)
            print 'Accuracy = {}'.format(acc)
        print 'proba mean score={}, std={}'.format(np.mean(cv_scores),np.std(cv_scores))
        print 'all accuracies----'
        print acc_scores
        print 'accuracy mean score={}, std={}'.format(np.mean(acc_scores),np.std(acc_scores))

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

class RandomForest_model(BaseModel):
    
    def set_classifier(self,**params):
        super(RandomForest_model,self).set_classifier()
        self.classifier = RandomForestClassifier(**params)
        
class GBM_model(BaseModel):    
    def set_classifier(self,**params):
        super(GBM_model,self).set_classifier()
        self.classifier = GradientBoostingClassifier(**params)
    def gridsearch_cv(self,param_list,verbose=1):
        if self.classifier is None:
            self.set_classifier()
        clf = model_selection.GridSearchCV(self.classifier,param_list,n_jobs=-1,scoring='neg_log_loss',verbose=verbose)
        clf.fit(self.X_train,self.y_train)
        print clf.cv_results_
        return clf.best_params_
        
class Logistic_model(BaseModel):
    
    def set_classifier(self,**params):
        super(Logistic_model,self).set_classifier()
        self.classifier = LogisticRegressionCV(**params)
    def get_feature(self,data_train,target_train,data_test):
        super(Logistic_model,self).get_feature(data_train,target_train,data_test)
        tmp = np.vstack((self.X_train,self.X_test))
        tmp = preprocessing.normalize(tmp)
        self.X_train = tmp[:self.X_train.shape[0],:]
        self.X_test = tmp[self.X_train.shape[0]:,:]
        
class SVM_model(BaseModel):
    
    def set_classifier(self,**params):
        super(SVM_model,self).set_classifier()
        self.classifier = svm.SVC(**params)
    def get_feature(self,data_train,target_train,data_test):
        super(SVM_model,self).get_feature(data_train,target_train,data_test)
        tmp = np.vstack((self.X_train,self.X_test))
        tmp = preprocessing.normalize(tmp)
        self.X_train = tmp[:self.X_train.shape[0],:]
        self.X_test = tmp[self.X_train.shape[0]:,:]

        
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
        super(XGB_model,self).reset_classifier()
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


class Stacking_model(BaseModel):
    def __init__(self,data_train,data_test,feature_list,target_train,classifier1,param1,classifier2,param2):
        super(Stacking_model,self).__init__(data_train,data_test,feature_list,target_train)
        self.clf1 = classifier1
        self.clf2 = classifier2
        self.param1 = param1
        self.param2 = param2
        
    def set_classifiers1(self,classifier1,**param):
        #used to pre-classification
        if self.clf1 is not None:
            del self.clf1
        if classifier1 == 'random forest':
            self.clf1 = RandomForestClassifier(**param)
        if classifier1 == 'gradient boosting':
            self.clf1 = GradientBoostingClassifier(**param)
        if classifier1 == 'extra trees':
            self.clf1 = ExtraTreesClassifier(**param)
        else:
            raise NameError("Invalid Classifiers")
    
            
    def set_classifier2(self,n_estimators=2800,
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
        super(XGB_model,self).reset_classifier()
        self.clf2 = XGBClassifier(n_estimators=n_estimators,
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
        
        
    def fit_predict_proba(self,nClass,nFold=9):    
        pred = np.zeros((self.X_test.shape[0],nClass))
        kfold = model_selection.StratifiedKFold(nFold,shuffle=True)
        for tr_inx,cv_inx in kfold.split(self.X_train,self.y_train):
        #import pdb;pdb.set_trace()
            if self.clf1 == 'random forest':
                clf1 = RandomForestClassifier(**self.param1)
            if self.clf1 == 'gradient boosting':
                clf1 = GradientBoostingClassifier(**self.param1)
            if self.clf1 == 'extra trees':
                clf1 = ExtraTreesClassifier(**self.param1)
            else:
                raise NameError("Invalid Classifiers one")
            
            clf1.fit(self.X_train[cv_inx,:],self.y_train[cv_inx])
            tr_pred = clf1.predict_proba(self.X_train[tr_inx,:])
            te_pred = clf1.predict_proba(self.X_test)
            X_train_ex = np.hstack((self.X_train[tr_inx,:],tr_pred))
            X_test_ex = np.hstack((self.X_test,te_pred))
            
            if self.clf2 == 'xgboost':
                clf2 = XGBClassifier(**self.param2)
            else:
                raise NameError('Invalide classifier 2')
                            
            clf2.fit(X_train_ex,self.y_train[tr_inx])#,num_rounds=2800)
            tmp = clf2.predict_proba(X_test_ex)
            pred += tmp
        pred = pred/nFold
        return pred