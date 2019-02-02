import pandas as pd
from flask import Flask
from joblib import load
from keras.models import load_model
import tensorflow as tf

app = Flask(__name__)

#################################################
# Create Routes
#################################################

rf = load('stroke_rf.joblib') 
svm_a = load('stroke_svm_grid_proba.joblib') 

nn_model = load_model("[13-14-7-1]_1000e_1b.h5")
graph = tf.get_default_graph()

pt = {'SEX':[0],'AGE':[50],'CIGPDAY':[0],'HEARTRTE':[85],'SYSBP':[120],'BPMEDS':[0],'TOTCHOL':[160],'BMI':[25],'GLUCOSE':[70],'DIABETES':[1],'EDUC_2.0':[0],'EDUC_3.0':[0],'EDUC_4.0':[1]}

@app.route("/")
def index():
	return "<a href='/random_forest'>/random_forest</a><br><a href='/svm'>/svm</a><br><a href='/neural_network'>/neural_network</a><br>"

@app.route("/random_forest")
def random_forest():

	good_patient = pd.DataFrame(pt)
	risk = rf.predict_proba(good_patient)[0,1]

	return f"Risk of stroke: {risk}<br><a href='/'>Home</a>"

@app.route("/svm")
def svm():

	good_patient = pd.DataFrame(pt)
	risk = svm_a.predict_proba(good_patient)[0,1]

	return f"Risk of stroke: {risk}<br><a href='/'>Home</a>"

@app.route("/neural_network")
def neural_network():
	global graph
	with graph.as_default():

		good_patient = pd.DataFrame(pt)
		risk = nn_model.predict_proba(good_patient)[0,0]

	return f"Risk of stroke: {risk}<br><a href='/'>Home</a>"

if __name__ == "__main__":
	app.run()