from collections import defaultdict
import pandas as pd
import numpy as np

class BaseFeatureExtraction(object):
	def __init__(self):
		self.method_info = ''
	def print_info(self):
		print(self.method_info)
	def get_info(self):
		return self.method_info
	def transform(self,X_train,X_test):
		return X_train,X_test

class CategoricalFeature(BaseFeatureExtraction):
    def __init__(self):
        super(CategoricalFeature , self ).__init__()