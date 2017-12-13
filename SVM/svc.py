import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score, learning_curve

digits = load_digits()

x = digits.data
y = digits.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

clf = SVC(C=10,kernel='linear',gamma=0.02)
clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

kf = KFold(n_splits=3, shuffle=True, random_state=42)
gamma_values = [0.1,0.05,0.02,0.01]
accuracy_scores = []

for gamma in gamma_values:
	clf = SVC (C=10,kernel='rbf',gamma=gamma)
	scores = cross_val_score(clf, x_train, y_train, cv=kf.split(x_train), scoring='accuracy')
	accuracy_score = scores.mean()
	accuracy_scores.append(accuracy_score)

best_index = np.array(accuracy_scores).argmax()
best_gamma = gamma_values[best_index]

clf = SVC(C=10, kernel='rbf', gamma=best_gamma)
clf.fit(x_train, y_train)

# y_pred = clf.predict(x_test)
# accuracy = metrics.accuracy_score(y_test, y_pred)
# report = metrics.classification_report(y_test,y_pred)
# print(accuracy)
# print(report)

train_sizes, train_scores, test_scores = learning_curve(clf, x_train, y_train, scoring='accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")

plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")

plt.legend()
plt.show()