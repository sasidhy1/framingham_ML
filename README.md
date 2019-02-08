# framingham_ML
A repository to store progress on ML project, compiled by Matt Lombardo, John Michals, Kenneth Reed, Yamini Sasidhar, Khrystyne Vaughan, Fardi Yeung.

This Framingham Risk Calculator has been deployed to Heroku as a Flask app [viewable here](https://framingham-ml-20186.herokuapp.com).

Additional data visualization is hosted on Tableau Public [viewable here](https://public.tableau.com/profile/jt7327#!/vizhome/Project3Dashboard_2/Dashboard1).

Available models: (scaled data indicated)
<p align="center">
  <img src="https://github.com/sasidhy1/framingham_ML/blob/master/images/model_diagram.svg" alt="Process Diagram"/>
</p>

## Repo Contents:
* PPT slides
* **application/**
    * (1) Flask application
    * (1) Procfile
    * (1) requirements.txt
    * **templates/**
      * (1) index.html
      * (1) results.html
    * **static/**
      * (1) style.css
      * (1) app.js
* **etl/**
    * (1) data cleanup jupyter ntbk
    * (1) CSV of reduced data
    * (2) python scripts
      * create feature scalers
      * convert .joblib -> .pkl
* **models/**
    * (1) TXT w/ R regression models (angina/CVD/stroke/diabetes)
    * (3) ANN KERAS models (angina/CVD/stroke)
    * (3) SVM SKLEARN models (angina/CVD/stroke)
    * (3) RF SKLEARN models (angina/CVD/stroke)
    * **scalers/**
      * (13) StandardScaler SKLEARN models (per feature)
* **training/**
    * (1) ANN jupyter ntbk (CVD)
    * (4) python scripts
      * (2) ANN scripts (angina/stroke)
      * (1) RF script (angina/CVD/stroke)
      * (1) SVM script (angina/CVD/stroke)
    * (1) regression R script (angina/CVD/stroke/diabetes)
