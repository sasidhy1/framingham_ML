import pandas as pd
import json
from flask import Flask, render_template, url_for, jsonify, request, redirect, json
import numpy as np
import sys
import keras
from keras.models import load_model
import tensorflow as tf
from joblib import load
import os

app = Flask(__name__)

# Model Holders
model = None

stroke_model = load_model("../models/framingham_stroke.h5")
cvd_model = load_model("../models/framingham_cvd.h5")
angina_model = load_model("../models/framingham_angina.h5")

graph = tf.get_default_graph()

#################################################
# Loading in the ../models
#################################################

@app.route("/")
def index():
	"""Return the homepage for health input."""
	return render_template("index.html")


@app.route("/about")
def about():
	"""Return information about our ../models"""
	return render_template("about.html")


@app.route('/testing',  methods=['POST'])
def test_model_prediction():

	gender = request.form['field1']
	age = request.form['field2']
	cigsperday = request.form['field3']
	systolic = request.form['field4']
	bpmeds = request.form['field5']
	education = request.form['field6']
	bmi = request.form['field7']
	heartrate = request.form['field8']
	glucose = request.form['field9']
	cholesterol = request.form['field10']
	diabetes = request.form['field11']

	if gender == 'male':
		sex = 0
	else:
		sex = 1

	if bpmeds == 'no':
		bp = 0
	else:
		bp = 1

	if diabetes == 'no':
		dia = 0
	else:
		dia = 1

	if education == "elemen":
		EDUC_2 = 0
		EDUC_3 = 0
		EDUC_4 = 0
	elif education == "highschool":
		EDUC_2 = 1
		EDUC_3 = 0
		EDUC_4 = 0
	elif education == "somecol":
		EDUC_2 = 0
		EDUC_3 = 1
		EDUC_4 = 0
	else:
		EDUC_2 = 0
		EDUC_3 = 0
		EDUC_4 = 1

	# store user data in dictionary
	user_data = {'SEX':[sex],'AGE':[age],'CIGPDAY':[cigsperday],'HEARTRTE':[heartrate],'SYSBP':[systolic],'BPMEDS':[bp],'TOTCHOL':[cholesterol],'BMI':[bmi],'GLUCOSE':[glucose],'DIABETES':[dia],'EDUC_2':[EDUC_2],'EDUC_3':[EDUC_3],'EDUC_4':[EDUC_4]}
	scaled_data = {}

	# create dictionary of scaled user data
	for key in user_data:
		if f'{key}_scaler.joblib' in os.listdir('../models/scalers'):
			scaler = load(f'../models/scalers/{key}_scaler.joblib')
			a = scaler.transform(user_data[key][0])
			scaled_data[key] = [a[0,0]]

	user_data_scaled = pd.DataFrame(scaled_data)

	# return ANN results
	nn_results = {}

	global graph
	with graph.as_default():
		nn_results['stroke'] = stroke_model.predict(user_data_scaled.drop(['EDUC_2','EDUC_3','EDUC_4'],axis=1))[0,0]
		nn_results['cvd'] = cvd_model.predict(user_data_scaled.drop(['EDUC_2','EDUC_3','EDUC_4','DIABETES','BPMEDS'],axis=1))[0,0]
		nn_results['angina'] = angina_model.predict(user_data_scaled.drop(['EDUC_2','EDUC_3','EDUC_4','DIABETES','BPMEDS','GLUCOSE','CIGPDAY'],axis=1))[0,0]

	stroke_results = nn_results['stroke']
	cvd_results = nn_results['cvd']
	angina_results = nn_results['angina']

	# format ANN results
	stroke_results = 100 * stroke_results
	cvd_results = 100 * cvd_results
	angina_results = 100 * angina_results

	stroke_results = "{0:.4f}%".format(stroke_results)
	cvd_results = "{0:.4f}%".format(cvd_results)
	angina_results = "{0:.4f}%".format(angina_results)

	# Regression Model Results
	reg_diabetes = -14.162026063 + ( 0.058979722 * float(age)) + ( 0.058252121 * float(bmi) ) + (0.051667194 * float(glucose)) + (0.009398841 * float(systolic)) - (0.379299003 * float(sex))
	reg_stroke = -7.918387257 + (0.057798562 * float(age)) + (0.012255363 * float(cigsperday)) - (0.002613120 * float(cholesterol)) + (0.023016992 * float(bmi)) - (0.005584217 * float(heartrate)) + ( 0.020128967 * float(systolic)) - ( 0.149489784 * float(sex))
	reg_angina = -5.679135058 + (0.025249146 * float(age)) + (0.005638028 * float(cholesterol)) + (0.031917873 * float(bmi)) - (0.006081201 * float(heartrate)) +  (0.008504367 * float(systolic)) - (0.548017154 * float(sex))
	reg_cvd = -7.165683847 + (0.044048379 * float(age)) + (0.009451579 * float(cigsperday)) + (0.003237465 * float(cholesterol)) + (0.031233422 * float(bmi)) + (0.004548587 * float(glucose)) - (0.005103114 * float(heartrate)) + (0.017607407 * float(systolic)) - (0.981634642 * float(sex))

	final_diabetes = 100 * (np.exp(reg_diabetes) / (1 + np.exp(reg_diabetes)))
	final_stroke = 100 * (np.exp(reg_stroke) / (1 + np.exp(reg_stroke)))
	final_angina = 100 * (np.exp(reg_angina) / (1 + np.exp(reg_angina)))
	final_cvd = 100 * (np.exp(reg_cvd) / (1 + np.exp(reg_cvd)))

	final_diabetes = "{0:.4f}%".format(final_diabetes)
	final_stroke = "{0:.4f}%".format(final_stroke)
	final_angina = "{0:.4f}%".format(final_angina)
	final_cvd = "{0:.4f}%".format(final_cvd)

	return render_template("results.html", stroke=stroke_results, cvd=cvd_results, angina=angina_results,diabetes=final_diabetes,regstroke=final_stroke,regangina=final_angina,regcvd=final_cvd)

if __name__ == "__main__":
	app.run()
