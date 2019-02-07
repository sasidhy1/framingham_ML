# import dependencies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# drop ID col
df = pd.read_csv('reduced.csv')
df = df.drop(['RANDID'],axis=1)
df.head()

# remove all responses, keep in target
response = 'STROKE'
dx = ['ANGINA','STROKE','CVD']
data = df.drop(dx, axis=1)

target = df[[response]]
print(data.shape, target.shape)

# get dummies for EDUC col, drop first
data = pd.get_dummies(data,columns=['EDUC'],drop_first=True)

# encode 1/2 SEX col to 0/1
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data['SEX'] = label_encoder.fit_transform(data['SEX'])

# create and save scalers
from joblib import dump
from sklearn.preprocessing import StandardScaler

pt = {'SEX':[1],'AGE':[65],'CIGPDAY':[15],'HEARTRTE':[80],'SYSBP':[135],
     'BPMEDS':[0],'TOTCHOL':[260],'BMI':[26],'GLUCOSE':[140],'DIABETES':[1],
     'EDUC_2.0':[1],'EDUC_3.0':[0],'EDUC_4.0':[0]}

for x,key in enumerate(data.columns.values):
    scaler = StandardScaler().fit(data[[key]])
    model_name = str(key).replace('.0','') + '_scaler.joblib'
    dump(scaler, f'../models/scalers/{model_name}')
    
# load and test scalers
import os
from joblib import load

pt = {'SEX':[1],'AGE':[65],'CIGPDAY':[15],'HEARTRTE':[80],'SYSBP':[135],
     'BPMEDS':[0],'TOTCHOL':[260],'BMI':[26],'GLUCOSE':[140],'DIABETES':[1],
     'EDUC_2':[1],'EDUC_3':[0],'EDUC_4':[0]}

scaled_pt = {}

try:
    for key in pt:
        if f'{key}_scaler.joblib' in os.listdir('../models/scalers'):
            scaler = load(f'../models/scalers/{key}_scaler.joblib')
            a = scaler.transform(pt[key][0])
            scaled_pt[key] = [a[0,0]]

except:
    print("error in loading model")