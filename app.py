import pandas as pd
import json
from flask import Flask, render_template, url_for, jsonify, request, redirect, json
import numpy as np
import sys
import keras
from keras.models import load_model
import tensorflow as tf


app = Flask(__name__)

# Model Holders
model = None

stroke_model = load_model("[13-14-7-1]_1000e_1b.h5")
graph = tf.get_default_graph()


#################################################
# Loading in the models
#################################################
# Loading a keras model with flask
# https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html


@app.route("/")
def index():
    """Return the homepage for health input."""
    return render_template("index.html")


@app.route("/about")
def about():
    """Return information about our models"""
    return render_template("about.html")


@app.route('/testing',  methods=['POST'])
def test_model_prediction():

    # data = json.loads(request.data)
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

    global graph
    with graph.as_default():
        pt = {'SEX':[sex],'AGE':[age],'CIGPDAY':[cigsperday],'HEARTRTE':[heartrate],'SYSBP':[systolic],'BPMEDS':[bp],'TOTCHOL':[cholesterol],'BMI':[bmi],'GLUCOSE':[glucose],'DIABETES':[dia],'EDUC_2.0':[EDUC_2],'EDUC_3.0':[EDUC_3],'EDUC_4.0':[EDUC_4]}
        good_patient = pd.DataFrame(pt)

        # Use the model to make a prediction
        results = stroke_model.predict_proba(good_patient)[0,0]
    
    print(results)

    # return f"Risk of stroke: {results}<br><a href='/'>resulsforheart</a>"
    return redirect(url_for('resultsforheart', stroke=results))


@app.route("/resultsforheart/<stroke>")
def resultsforheart(stroke):
    """Return the predictions of heart conditions."""
    return render_template("results.html", stroke=stroke)


@app.route("/diabetes")
def compare():
    """Return the prediction for diabetes."""
    return render_template("diabetes.html")


if __name__ == "__main__":
    app.run()
