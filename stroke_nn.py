#%% import dependencies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%% drop ID col
df = pd.read_csv('../reduced.csv')
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

#%% split data 80/20 train/test, stratify target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data,target,random_state=42,test_size=0.20,stratify=target)

#%% scale and transform variable data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.fit_transform(X_test)

#%% avoid batch for small dataset, better density gradient
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=14, activation='relu', input_dim=13))
model.add(Dense(units=7, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=1000,
    shuffle=True,
    verbose=2
)

#%% view training/testing accuracy, should be similar to avoid overfitting
model_loss, model_accuracy = model.evaluate(
    X_train_scaled, y_train, verbose=2)
print(f"Training - Loss: {model_loss}, Accuracy: {model_accuracy}")

model_loss, model_accuracy = model.evaluate(
    X_test_scaled, y_test, verbose=2)
print(f"Testing - Loss: {model_loss}, Accuracy: {model_accuracy}")

#%% accuracy should increase over epochs
plt.plot(history.history['acc'])
plt.title('Training Accuracy over 1000 Iterations')
plt.ylabel('Training Accuracy')
plt.xlabel('Iterations')
plt.grid(axis='y')
#plt.savefig('[13-14-7-1]_acc_1000_1.png')

#%% error should decrease over epochs, steeper the better
plt.plot(history.history['loss'])
plt.title('Training with Backpropagation over 1000 Iterations')
plt.ylabel('Training Loss')
plt.xlabel('Iterations')
plt.grid(axis='y')
#plt.savefig('[13-14-7-1]_loss_1000_1.png')

#%% view how correct and incorrect guess count
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

#%% create a sample "good" patient, not from data
pt = {'SEX':[0],'AGE':[50],'CIGPDAY':[0],'HEARTRTE':[90],'SYSBP':[120],
     'BPMEDS':[0],'TOTCHOL':[160],'BMI':[26.5],'GLUCOSE':[95],'DIABETES':[0],
     'EDUC_2.0':[0],'EDUC_3.0':[0],'EDUC_4.0':[1]}
good_patient = pd.DataFrame(pt)

# return probability of response (stroke)
model.predict_proba(good_patient)

#%% create a sample "bad" patient, not from data
pt = {'SEX':[1],'AGE':[80],'CIGPDAY':[10],'HEARTRTE':[90],'SYSBP':[120],'BPMEDS':[1],'TOTCHOL':[160],'BMI':[30],'GLUCOSE':[95],'DIABETES':[1],'EDUC_2.0':[0],'EDUC_3.0':[0],'EDUC_4.0':[1]}
bad_patient = pd.DataFrame(pt)

# return probability of response (stroke)
model.predict_proba(bad_patient)