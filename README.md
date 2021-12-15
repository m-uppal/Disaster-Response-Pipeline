# Disaster-Response-Pipeline

## Table of Contents

1. Project Motivation
2. Libraries Used
3. Installation
4. File Descriptions
5. Summary of Findings
6. Acknowledgements

## Project Motivation/intro

The project goal is to build a web application utilizing a dataset containing pre-labelled tweets and messages from real-life disaster events (eg. flood, earthquake)in combination with a Natural Language Processing (NLP) model to classify messages in real time that in turn can directed to appropriate aid organizations.

This project is divided into the following sections:

- Data Processing: building an ETL pipeline to extract data from dataset, clean data and save in a SQLite database
- Machine Learning Pipeline: build a machine learning pipeline to predict message categories
- Web development: Run a web application that categorizes message in real time 

## Libraries Used/installation

- Python 3+
- Machine Learning Libraries:NumPy, SciPy, Pandas, Scikit-Learn
- Natural Language Processing Libraries:NLTK
- SQLite Database Librarires: SQLalchemy
- Model Loading and Saving Library: Pickle
- Web App and Data Visualisation: Flask, Plotly

## Instructions:
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## File Descriptions

- data: dataset provided by Figure Eight, 2 csv files comprised of of disaster response messages and message categories 
- ETL Preparation Notebook:
- ML Pipeline Preparation Notebook:
- app:

## Summary of Findings/screenshots/results

## Acknowledgements

This web application is part of the Udacity Data Science Nanodegree Program, the dataset is provided by Figure Eight. 


