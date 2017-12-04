#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score, learning_curve
from sklearn.model_selection import GridSearchCV

digits = load_digits()

x = digits.data
y = digits.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

possible_parameters = {
    'C': [1e0, 1e1, 1e2, 1e3],
    'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
}

svc = SVC(kernel='rbf')

# The GridSearchCV is itself a classifier
# we fit the GridSearchCV with the training data
# and then we use it to predict on the test set
clf = GridSearchCV(svc, possible_parameters, n_jobs=4) # n_jobs=4 means we parallelize the search over 4 threads
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
report = metrics.classification_report(y_test,y_pred)
cm = metrics.confusion_matrix(y_test,y_pred)
print(accuracy)
print(report)
print(cm)
print(clf.best_params_)