# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 10:40:06 2019

@author: fardi

Rutgers Data Science Camp Final Project - Machine Learning

Data from Framingham heart disease studies.
'SEX','AGE','TOTCHOL','BMI','HEARTRTE','SYSBP'
"""



# Importing the dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# setting the predicting parameters X and target respnose y (Angina)
dataset = pd.read_csv('max_reduce.csv')
X = dataset[['SEX','AGE','TOTCHOL','BMI','HEARTRTE','SYSBP']]
X = X.iloc[:, 0:6].values
#X = X.iloc[:, [0,1,2,4,6,7,8,9,10,12]].values
y = dataset.iloc[:, 13].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# encode sex values from 1 and 2 or 1 or 0
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
labelencoder_X_1 = LabelEncoder()
X[:,10] = labelencoder_X_1.fit_transform(X[:, 10])
onehotencoder = OneHotEncoder(categorical_features = [10])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set with stratify on target response y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 2, init = 'uniform', activation = 'relu', input_dim = 6))
#classifier.add(Dense(output_dim = 2, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
history = classifier.fit(X_train, y_train, batch_size = 32, nb_epoch = 100)


# Predicting the Test set results - setting the probability to 80%
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.80)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#cm accuracy = 0.8367

#saving model
classifier.save("framingham_angina.h5")


import json
with open('history_Angina.json', 'w') as f:
    json.dump(history.history, f)


plt.plot(history.history['acc'])
plt.title('Training Accuracy over 100 Iterations')
plt.ylabel('Training Accuracy')
plt.xlabel('Iterations')
plt.grid(axis='y')


plt.plot(history.history['loss'])
plt.title('Training with Backpropagation over 100 Iterations')
plt.ylabel('Training Loss')
plt.xlabel('Iterations')
plt.grid(axis='y')

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu', input_dim = 13))
    classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [32,50],
              'epochs': [50,100, 200],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
{'batch_size': 32, 'epochs': 50, 'optimizer': 'adam'}
best_accuracy = grid_search.best_score_ 
#0.836734693877551

#'SEX','AGE','TOTCHOL','BMI','HEARTRTE','SYSBP'
# create a sample "bad" patient, not from data
pt_bad = {'SEX':[1],'AGE':[90], 'TOTCHOL':[500],'BMI':[60],'HEARTRTE':[150],'SYSBP':[330]}
bad_patient = pd.DataFrame(pt_bad)

# return probability of response (cvd)
classifier.predict_proba(bad_patient)


pt_good = {'SEX':[0],'AGE':[20], 'TOTCHOL':[100],'BMI':[18],'HEARTRTE':[60],'SYSBP':[80]}
good_patient = pd.DataFrame(pt_good)

# return probability of response (cvd)
classifier.predict_proba(good_patient)

##################### 1 hidden layer with 2 nodes, 100 

pt_bad = {'SEX':[1],'AGE':[90], 'TOTCHOL':[500],'BMI':[60],'HEARTRTE':[150],'SYSBP':[330]}
bad_patient = pd.DataFrame(pt_bad)

# return probability of response (cvd)
classifier.predict_proba(bad_patient)
Out[108]: array([[0.34387156]], dtype=float32)

pt_good = {'SEX':[0],'AGE':[20], 'TOTCHOL':[100],'BMI':[18],'HEARTRTE':[60],'SYSBP':[80]}
good_patient = pd.DataFrame(pt_good)

# return probability of response (cvd)
classifier.predict_proba(good_patient)
Out[109]: array([[0.04523359]], dtype=float32)


##################### 1 hidden layer with 2 nodes, epochs = 50

pt_bad = {'SEX':[1],'AGE':[90], 'TOTCHOL':[500],'BMI':[60],'HEARTRTE':[150],'SYSBP':[330]}
bad_patient = pd.DataFrame(pt_bad)

# return probability of response (cvd)
classifier.predict_proba(bad_patient)
Out[114]: array([[0.3238456]], dtype=float32)

pt_good = {'SEX':[0],'AGE':[20], 'TOTCHOL':[100],'BMI':[18],'HEARTRTE':[60],'SYSBP':[80]}
good_patient = pd.DataFrame(pt_good)

# return probability of response (cvd)
classifier.predict_proba(good_patient)
Out[115]: array([[0.3238456]], dtype=float32)
################   predictive power is not sensitive


##################### 1 hidden layer with 1 nodes, epochs = 100
pt_bad = {'SEX':[1],'AGE':[90], 'TOTCHOL':[500],'BMI':[60],'HEARTRTE':[150],'SYSBP':[330]}
bad_patient = pd.DataFrame(pt_bad)

# return probability of response (cvd)
classifier.predict_proba(bad_patient)
Out[117]: array([[0.29326728]], dtype=float32)

pt_good = {'SEX':[0],'AGE':[20], 'TOTCHOL':[100],'BMI':[18],'HEARTRTE':[60],'SYSBP':[80]}
good_patient = pd.DataFrame(pt_good)

# return probability of response (cvd)
classifier.predict_proba(good_patient)
Out[118]: array([[0.29326728]], dtype=float32)
################   predictive power is also not sensitive

