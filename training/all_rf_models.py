# import dependencies
import pandas as pd
import numpy as np

# drop ID col
df = pd.read_csv('reduced.csv')

# DROP EDUC FOR CVD
df = df.drop(['RANDID','EDUC'],axis=1)
df.head()

# REPLACE FOR DIFFERENT MODELS
response = 'CVD'
dx = ['ANGINA','STROKE','CVD']

# remove all responses, keep in target
data = df.drop(dx, axis=1)

target = df[[response]]
print(data.shape, target.shape)

# get dummies for EDUC col, drop first
try:
    data = pd.get_dummies(data,columns=['EDUC'],drop_first=True)
except:
    print('no EDUC column')

# encode 1/2 SEX col to 0/1
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data['SEX'] = label_encoder.fit_transform(data['SEX'])
feature_names = data.columns

# split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=42)

# train model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500,min_samples_leaf=100)
rf = rf.fit(X_train, y_train.values.ravel())

# view feature importances
sorted(zip(rf.feature_importances_, feature_names), reverse=True)

from joblib import dump
#dump(rf, 'models/filename.joblib')

from joblib import load
# rf = load('models/filename.joblib') 

# print accuracies
print('Training Acc: %.3f' % rf.score(X_train1, y_train))
print('Testing Acc: %.3f' % rf.score(X_test1, y_test))

# create a sample "good" patient, not from data
pt = {'SEX':[0],'AGE':[50],'CIGPDAY':[0],'HEARTRTE':[85],'SYSBP':[120],
     'BPMEDS':[0],'TOTCHOL':[160],'BMI':[25],'GLUCOSE':[70],'DIABETES':[1],
     'EDUC_2.0':[0],'EDUC_3.0':[0],'EDUC_4.0':[1]}

good_patient = pd.DataFrame(pt)
rf.predict_proba(good_patient)[0,1]

# create a sample "bad" patient, not from data
pt = {'SEX':[1],'AGE':[85],'CIGPDAY':[15],'HEARTRTE':[90],'SYSBP':[220],
     'BPMEDS':[1],'TOTCHOL':[210],'BMI':[40],'GLUCOSE':[200],'DIABETES':[1],
     'EDUC_2.0':[0],'EDUC_3.0':[0],'EDUC_4.0':[0]}

bad_patient = pd.DataFrame(pt)
rf.predict_proba(bad_patient)[0,1]
