# import dependencies
import os
import pandas as pd
from joblib import load
from sklearn.externals import joblib

# load each RF joblib model, dump as pkl
for file in os.listdir('models/'):
    if '_rf.joblib' in file:
        model = load(f'models/{file}')
        filename = file.replace('.joblib','.pkl')
        joblib.dump(model, f'models/{filename}')

# load each SVM joblib model, dump as pkl
for file in os.listdir('models/'):
    if '_svm.joblib' in file:
        model = load(f'models/{file}')
        filename = file.replace('.joblib','.pkl')
        joblib.dump(model, f'models/{filename}')

# load and test one model
try:
    rf = joblib.load('models/cvd_svm.pkl')

    # create a sample "good" patient, not from data
    pt = {'SEX':[0],'AGE':[50],'CIGPDAY':[0],'HEARTRTE':[85],'SYSBP':[120],
         'BPMEDS':[0],'TOTCHOL':[160],'BMI':[25],'GLUCOSE':[70],'DIABETES':[1],
         'EDUC_2.0':[0],'EDUC_3.0':[0],'EDUC_4.0':[1]}

    good_patient = pd.DataFrame(pt)
    rf.predict_proba(good_patient)[0,1]

except:
    print("error in loading model")