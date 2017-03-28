
from FeatureExtraction import *

class BaseModel:
	def __init__(self,feature_list,classifier,data_train,label_train,data_test):
		self.feature_list = feature_list
		self.classifier = classifier
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.get_feature(data_train,label_train,data_test)


	def get_feature(self,data_train,label_train,data_test):		
		self.X_train = data_train.copy()
		self.y_train = label_train.copy()
		self.X_test = data_test.copy()
		for ff in feature_list:
			self.X_train,self.X_test = ff.transform(self.X_train,self.X_test,self.y_train)

	def cv_study(self):
		pass

	def fit_predict(self,return_proba=False):
		self.classifier.fit(self.X_train,self.y_train)
		if return_proba:
			return self.classifier.predict_proba(self.X_test)
		else:
			return self.classifier.predict(self.X_test)
