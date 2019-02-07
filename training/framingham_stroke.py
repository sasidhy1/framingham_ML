# import dependencies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# drop ID col
df = pd.read_csv('reduced.csv')
df = df.drop(['RANDID','EDUC'],axis=1)
df.head()

# remove all responses, keep in target
response = 'STROKE'
dx = ['ANGINA','STROKE','CVD']
data = df.drop(dx, axis=1)

target = df[[response]]
print(data.shape, target.shape)

# get dummies for EDUC col, drop first
try:
    data = pd.get_dummies(data,columns=['EDUC'],drop_first=True)
except:
    print("no EDUC column")

# encode 1/2 SEX col to 0/1
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data['SEX'] = label_encoder.fit_transform(data['SEX'])

# split data 80/20 train/test, stratify target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data,target,random_state=42,stratify=target)

# scale and transform variable data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.fit_transform(X_test)

# avoid batch for small dataset, better density gradient
from keras.models import Sequential
from keras.layers import Dense

dim = data.shape[1]

model = Sequential()
model.add(Dense(units=dim+1, activation='relu', input_dim=dim))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=20,
    batch_size=35,
    validation_data=(X_test_scaled,y_test),
    shuffle=True,
    verbose=2
)

# view training/testing accuracy, should be similar to avoid overfitting
model_loss, model_accuracy = model.evaluate(
    X_train_scaled, y_train, verbose=2)
print(f"Training - Loss: {model_loss}, Accuracy: {model_accuracy}")

model_loss, model_accuracy = model.evaluate(
    X_test_scaled, y_test, verbose=2)
print(f"Testing - Loss: {model_loss}, Accuracy: {model_accuracy}")

# view how correct and incorrect guess count
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

predictions = model.predict_classes(X_test_scaled)
test_df = pd.DataFrame({'predicted':np.ravel(predictions),'actual':np.ravel(y_test)})
pred_count(test_df)

# accuracy should increase over epochs
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Training Accuracy over 100 Iterations')
plt.ylabel('Training Accuracy')
plt.xlabel('Iterations')
plt.legend(['Training Accuracy','Validation Accuracy'])
plt.grid(axis='y')

# error should decrease over epochs, steeper the better
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss over 100 Iterations')
plt.ylabel('Training Loss')
plt.xlabel('Iterations')
plt.legend(['Training Loss','Validation Loss'])
plt.grid(axis='y')

#model.save('models/filename.h5')

from keras.models import load_model
# model = load_model("models/filename.h5")

# create a sample "good" patient, not from data
def reduce_pt(pt,data):
    to_del = [x for x in pt if x not in data.columns.values]

    for x in to_del:
        del pt[x]
        
    return pt

from joblib import load

# scale new data
def scale_features(pt):
    scaled_pt = {}

    for key in pt:
        if f'{key}_scaler.joblib' in os.listdir('models/scalers'):
            scaler = load(f'models/scalers/{key}_scaler.joblib')
            val = scaler.transform(pt[key][0])
            scaled_pt[key] = [val[0,0]]
            
    return scaled_pt

pt = {'SEX':[1],'AGE':[50],'CIGPDAY':[0],'HEARTRTE':[100],'SYSBP':[130],
     'BPMEDS':[1],'TOTCHOL':[240],'BMI':[25],'GLUCOSE':[100],'DIABETES':[0],
     'EDUC_2.0':[0],'EDUC_3.0':[1],'EDUC_4.0':[0]}

pt = reduce_pt(pt,data)
pt_scaled = scale_features(pt)
good_patient = pd.DataFrame(pt_scaled)

model.predict_proba(good_patient)[0,0]

# create a sample "bad" patient, not from data
pt = {'SEX':[1],'AGE':[65],'CIGPDAY':[90],'HEARTRTE':[200],'SYSBP':[135],
     'BPMEDS':[0],'TOTCHOL':[400],'BMI':[26],'GLUCOSE':[300],'DIABETES':[1],
     'EDUC_2.0':[1],'EDUC_3.0':[0],'EDUC_4.0':[0]}

pt = reduce_pt(pt,data)
pt_scaled = scale_features(pt)
bad_patient = pd.DataFrame(pt_scaled)
 
# return probability of response (stroke)
model.predict_proba(bad_patient)[0,0]