#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from time import time


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]


#########################################################
### your code goes here ###
svc = SVC(kernel='rbf', C=10000)
start = time()
print('Start training the training Data...')
svc.fit(features_train, labels_train)
print('Finished training.')
print('time spent training: ', round(time()-start, 3))

print('Start Prediciting test data...')

start = time()
pred = svc.predict(features_test)
print('time spent prediction: ',round(time()-start, 3))

print('Check accuracy score')
print(accuracy_score(pred, labels_test))



#########################################################


