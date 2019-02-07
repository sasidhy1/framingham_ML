# framingham_ML
A repository to store progress on ML project, compiled by Matt Lombardo, John Michals, Kenneth Reed, Yamini Sasidhar, Khrystyne Vaughan, Fardi Yeung.

This Framingham Risk Calculator has been deployed to Heroku as a Flask app [viewable here](https://hidden-island-65494.herokuapp.com).

Available models: (scaled data indicated)
<p align="center">
  <img src="https://github.com/sasidhy1/framingham_ML/blob/master/images/model_diagram.svg" alt="Process Diagram"/>
</p>

## Repo Contents:
* PPT slides
* **calculator_app/**
    * (1) Flask application
    * (2) HTML templates
    * (1) static CSS sheet
    * (1) static JS script
* **models/**
    * (1) TXT w/ R regression models (angina/CVD/stroke/diabetes)
    * (3) ANN KERAS models (angina/CVD/stroke)
    * (3) SVM SKLEARN models (angina/CVD/stroke)
    * (3) RF SKLEARN models (angina/CVD/stroke)
    * (13) StandardScaler SKLEARN models (per feature)
* **training/**
    * (1) ETL jupyter ntbk
    * (1) ANN jupyter ntbk (CVD)
    * (2) ANN python scripts (angina/stroke)
    * (1) RF python script (angina/CVD/stroke)
    * (1) SVM python script (angina/CVD/stroke)
    * (1) R regression script (angina/CVD/stroke/diabetes)
