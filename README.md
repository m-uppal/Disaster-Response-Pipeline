# Disaster Response Pipeline

## Table of Contents

1. Purpose
2. Libraries Used
3. Insructions
4. File Descriptions
5. Results
6. Acknowledgements

## Purpose

The project goal is to build a Flask web application utilizing a dataset containing pre-labelled tweets and messages from real-life disaster events (eg. flood, earthquake)in combination with a Natural Language Processing (NLP) model to classify messages in real time that in turn can directed to appropriate aid organizations.

This project is divided into the following sections:

- Data Processing: building an ETL pipeline to extract data from dataset, clean data and save in a SQLite database
- Machine Learning Pipeline: build a machine learning pipeline to predict message categories
- Web development: Run a web application that categorizes messages in real time 

## Libraries Used

- Python 3+
- Machine Learning Libraries: NumPy, SciPy, Pandas, Scikit-Learn
- Natural Language Processing Libraries: NLTK
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
- process_data.py: ETL pipeline used for data cleaning, feature extraction, and storing data in SQLite database
- train_classifer.py: machine learning pipeline that loads data, trains model, saves trained model as a pickle file. 
- run.py: launches Flask web app used to classify messages

## Results

### Example of message entered in app: 

![Message to classify](https://user-images.githubusercontent.com/42184785/146237896-a46df0ae-49e6-4945-9dc4-a9a52d785298.png)


### Categories that messages classified as displayed as highlighted in green:

![Message classification result](https://user-images.githubusercontent.com/42184785/146237754-cf67f39b-0bb9-47c4-8966-71eb91b81938.png)


### Visualizations of training dataset:

![Visualization 1](https://user-images.githubusercontent.com/42184785/146237040-18baf42c-ea56-4e34-bb51-050b1e17f5f7.png)

![Visualization 2](https://user-images.githubusercontent.com/42184785/146237060-22355015-2ded-4e25-8c16-b95710d49c28.png)


### Model results:

![Model Results](https://user-images.githubusercontent.com/42184785/146236947-4865a4ac-7554-4123-ae39-f8465e429e4c.png)


## Acknowledgements

This web application is part of the Udacity Data Science Nanodegree Program, the dataset is provided by Figure Eight. 


