# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 11:51:28 2018

@author: Shivam-PC
"""

import numpy as np
import pandas as pd

voice_data = pd.read_csv("voice.csv")


x = voice_data.iloc[:,:-1].values
y = voice_data.iloc[:,20].values
y = pd.DataFrame(y)
y = pd.get_dummies(y,drop_first=True)

#spitting the dataset into training set and text set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split( x , y, test_size=0.2, random_state=42)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here
# LogisticRegression accuracy - 0.971 percent
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
print(classifier.score(X_test, y_test))

# Decision tree accuracy 0.963 percent
from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf = clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

# GaussianNB accuracy 0.895
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

#  AdaBoostClassifier accuracy 0.981
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
clf = AdaBoostClassifier()
clf.fit(X_train, y_train) 
print(clf.score(X_test, y_test))

# RandomForestClassifier accuracy 0.981
clf = RandomForestClassifier()
clf.fit(X_train, y_train) 
print(clf.score(X_test, y_test))

# svm accuracy 0.982
from sklearn import svm
clf = svm.SVC(kernel='rbf', C=1)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(classifier.score(X_test,y_test))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
