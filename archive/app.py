import pandas as pd
import json
from flask import Flask, render_template, abort, jsonify, request, redirect, json
import numpy as np
import sys
import keras
from keras.models import load_model
import tensorflow as tf


app = Flask(__name__)

# Model Holders
model = None
graph = None

stroke_model = load_model("[13-14-7-1]_1000e_1b.h5")
graph = tf.get_default_graph()


#################################################
# Loading in the models
#################################################
# Loading a keras model with flask
# https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html


# @app.route("/")
# def index():
#     """Return the homepage for health input."""
#     return render_template("index.html")


@app.route("/about")
def about():
    """Return information about our models"""
    return render_template("about.html")


@app.route('/')
def test_model_prediction():
    # data = json.loads(request.data)
    # for k, v in data.items():
    #     print("This is the key: " + k )
    #     print("This is the value: " + str(v))


global graph
with graph.as_default():
    pt = {'SEX': [0], 'AGE': [50], 'CIGPDAY': [0], 'HEARTRTE': [85], 'SYSBP': [120], 'BPMEDS': [0], 'TOTCHOL': [
        160], 'BMI': [25], 'GLUCOSE': [70], 'DIABETES': [1], 'EDUC_2.0': [0], 'EDUC_3.0': [0], 'EDUC_4.0': [1]}

    good_patient = pd.DataFrame(pt)
    predicted = stroke_model.predict_proba(good_patient)[0, 0]

    # Use the model to make a prediction
    # predicted_results = model.predict_proba(data)
    print(predicted)

    return f"Risk of stroke: {predicted}<br><a href='/'>Home</a>"


@app.route("/resultsforheart")
def resultsforheart():
    """Return the predictions of heart conditions."""
    return render_template("resultsforheart.html")


@app.route("/diabetes")
def compare():
    """Return the prediction for diabetes."""
    return render_template("diabetes.html")


if __name__ == "__main__":
    app.run()
