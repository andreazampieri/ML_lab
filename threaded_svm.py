import matplotlib.pyplot as plt
import numpy as np
import threading
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score

class tSVC(threading.Thread):

	def __init__ (self,threadID,name,classifier,data,labels):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.name = name
		self.ending_lock = threading.Lock()
		self.ending_lock.acquire()
		self.data = data
		self.labels = labels
		self.classifier = classifier

	def run(self):
		self.classifier.fit(self.data,self.labels)
		self.ending_lock.release()

	def predict(self,test_data):
		return self.classifier.predict(test_data)

	