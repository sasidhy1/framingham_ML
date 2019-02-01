#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% drop ID col
df = pd.read_csv('reduced.csv')
df = df.drop(['RANDID'],axis=1)
df.head()

#%% remove all responses, keep in target
response = 'STROKE'
dx = ['ANGINA','STROKE','CVD']
data = df.drop(dx, axis=1)

target = df[[response]]
print(data.shape, target.shape)

#%% get dummies for EDUC col, drop first
data = pd.get_dummies(data,columns=['EDUC'],drop_first=True)

#%% encode 1/2 SEX col to 0/1
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data['SEX'] = label_encoder.fit_transform(data['SEX'])

#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=42)

#%% scale and transform variable data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.fit_transform(X_test)

#%%
# Fit to the training data and validate with the test data
from sklearn.svm import SVC 
model = SVC(kernel='linear',probability=True)

#%%
# Create the GridSearch estimator along with a parameter object containing the values to adjust
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [1, 5, 10, 50],
              'gamma': [0.0001, 0.0005, 0.001, 0.005]}

grid = GridSearchCV(model, param_grid, verbose=3)

#%%
grid.fit(X_train_scaled, y_train.values.ravel())

import winsound
duration = 1000  # millisecond
freq = 550  # Hz
winsound.Beep(freq, duration)

#%% best parameters for dataset
print(grid.best_params_)    #{'C': 1, 'gamma': 0.0001}
print(grid.best_score_)

#%%
from joblib import dump
#dump(grid, 'models/stroke_svm_grid_proba.joblib')

from joblib import load
#model = load('filename.joblib')  

#%%
def pred_count(df):
    corr = 0
    wron = 0
    for index, row in df.iterrows():
        if row['predicted'] == row['actual']:
            corr = corr + 1
        else:
            wron = wron + 1
            
    print(f'Correct predictions: {corr}')
    print(f'Incorrect predictions: {wron}')

predictions = grid.predict(X_test_scaled)
test_df = pd.DataFrame({'predicted':np.ravel(predictions),'actual':np.ravel(y_test)})
pred_count(test_df)

#%%
print('Training Acc: %.3f' % grid.score(X_train_scaled, y_train))
print('Testing Acc: %.3f' % grid.score(X_test_scaled, y_test))

#%%
# Calculate classification report
from sklearn.metrics import classification_report
predictions = grid.predict(X_test_scaled)
print(classification_report(y_test, predictions,
                            target_names=["negative", "positive"]))

#%% create a sample "good" patient, not from data
pt = {'SEX':[0],'AGE':[60],'CIGPDAY':[0],'HEARTRTE':[60],'SYSBP':[120],
     'BPMEDS':[1],'TOTCHOL':[100],'BMI':[25],'GLUCOSE':[70],'DIABETES':[0],
     'EDUC_2.0':[0],'EDUC_3.0':[0],'EDUC_4.0':[1]}
good_patient = pd.DataFrame(pt)

# return probability of response (stroke)
#grid.predict(good_patient)
grid.predict_proba(good_patient)[0,1]

#%% create a sample "bad" patient, not from data
pt = {'SEX':[1],'AGE':[75],'CIGPDAY':[5],'HEARTRTE':[68],'SYSBP':[200],
     'BPMEDS':[1],'TOTCHOL':[210],'BMI':[25],'GLUCOSE':[70],'DIABETES':[0],
     'EDUC_2.0':[0],'EDUC_3.0':[0],'EDUC_4.0':[1]}
bad_patient = pd.DataFrame(pt)

# return probability of response (stroke)
#grid.predict(bad_patient)
grid.predict_proba(bad_patient)[0,1]
