#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# print(len(features_train[0]))


#########################################################
### your code goes here ###

start = time()
print('Start training the Decision Tree Classifier...')

adb = AdaBoostClassifier(n_estimators=500)


adb.fit(features_train, labels_train)

print('time spent: ', round(time()-start, 3))

start = time()
print('Start prediction based on Decision Tree Classifier...')

pred = adb.predict(features_test)
print('time spent for prediction: ', round(time()-start, 3))

pred_accuracy = accuracy_score(pred, labels_test)

print('Accuracy score: ', pred_accuracy)

#########################################################


