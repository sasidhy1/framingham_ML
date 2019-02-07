import pandas as pd
from flask import Flask
from joblib import load
from sklearn.externals import joblib
from keras.models import load_model
import tensorflow as tf
import os

app = Flask(__name__)

#################################################
# Create Routes
#################################################

nn_angina = load_model("models/framingham_angina.h5")
nn_cvd = load_model("models/framingham_cvd.h5")
nn_stroke = load_model("models/framingham_stroke.h5")

graph = tf.get_default_graph()

pt = {'SEX':[0],'AGE':[50],'CIGPDAY':[0],'HEARTRTE':[85],'SYSBP':[120],'BPMEDS':[0],'TOTCHOL':[160],'BMI':[25],'GLUCOSE':[70],'DIABETES':[1],'EDUC_2':[0],'EDUC_3':[0],'EDUC_4':[1]}

risk_rf = 0
risk_svm = 0
risk_nn = 0

scaled_pt = {}

for key in pt:
    if f'{key}_scaler.joblib' in os.listdir('models/scalers'):
        scaler = load(f'models/scalers/{key}_scaler.joblib')
        a = scaler.transform(pt[key][0])
        scaled_pt[key] = [a[0,0]]

@app.route("/")
def index():
	return "<a href='/angina_results'>Angina Results</a><br><a href='/cvd_results'>CVD Results</a><br><a href='/stroke_results'>Stroke Results</a>"

@app.route("/angina_results")
def angina_results():

	good_patient = pd.DataFrame(pt)
	good_patient_scaled = pd.DataFrame(scaled_pt)

 	# RF -- NOT SCALED
	rf_angina = joblib.load('models/angina_rf.pkl')
	risk_rf = rf_angina.predict_proba(good_patient)[0,1]
	
 	# SVM -- SCALED
	svm_angina = joblib.load('models/angina_svm.pkl') 
	risk_svm = svm_angina.predict_proba(good_patient_scaled)[0,1]

 	# NN -- SCALED
	global graph
	with graph.as_default():
		risk_nn = nn_angina.predict_proba(good_patient_scaled.drop(['EDUC_2','EDUC_3','EDUC_4','DIABETES','BPMEDS','GLUCOSE','CIGPDAY'],axis=1))[0,0]

	return f"Risk of Angina (RF): {risk_rf}<br>Risk of Angina (SVM): {risk_svm}<br>Risk of Angina (NN): {risk_nn}<br><a href='/'>Home</a>"

@app.route("/cvd_results")
def cvd_results():

	good_patient = pd.DataFrame(pt)
	good_patient_scaled = pd.DataFrame(scaled_pt)

	# RF -- NOT SCALED
	rf_cvd = joblib.load('models/cvd_rf.pkl') 
	risk_rf = rf_cvd.predict_proba(good_patient.drop(['EDUC_2','EDUC_3','EDUC_4'],axis=1))[0,1]
	
	# SVM -- SCALED
	svm_cvd = joblib.load('models/cvd_svm.pkl')
	risk_svm = svm_cvd.predict_proba(good_patient_scaled)[0,1]

	# NN -- SCALED
	global graph
	with graph.as_default():
		risk_nn = nn_cvd.predict_proba(good_patient_scaled.drop(['EDUC_2','EDUC_3','EDUC_4','DIABETES','BPMEDS'],axis=1))[0,0]

	return f"Risk of CVD (RF): {risk_rf}<br>Risk of CVD (SVM): {risk_svm}<br>Risk of CVD (NN): {risk_nn}<br><a href='/'>Home</a>"

@app.route("/stroke_results")
def stroke_results():

	good_patient = pd.DataFrame(pt)
	good_patient_scaled = pd.DataFrame(scaled_pt)

	# RF -- NOT SCALED
	rf_stroke = joblib.load('models/stroke_rf.pkl') 
	risk_rf = rf_stroke.predict_proba(good_patient)[0,1]
	
	# SVM -- SCALED
	svm_stroke = joblib.load('models/stroke_svm.pkl') 
	risk_svm = svm_stroke.predict_proba(good_patient_scaled)[0,1]

	# NN -- SCALED
	global graph
	with graph.as_default():
		risk_nn = nn_stroke.predict_proba(good_patient_scaled.drop(['EDUC_2','EDUC_3','EDUC_4'],axis=1))[0,0]

	return f"Risk of Stroke (RF): {risk_rf}<br>Risk of Stroke (SVM): {risk_svm}<br>Risk of Stroke (NN): {risk_nn}<br><a href='/'>Home</a>"

if __name__ == "__main__":
	app.run()