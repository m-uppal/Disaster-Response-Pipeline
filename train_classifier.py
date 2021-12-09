# import packages
import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

def load_data(data_file):
    """
    Function loads data from database
    
    Arguments:
        data_file -> Path to SQLite database containing cleaned dataset
    Output:
        X -> dataframe containing features
        Y -> dataframe containing labels
        category_names -> list of category names
    """
    # read in file
    engine = create_engine('sqlite:///' + data_file)

    # load to database
    df = pd.read_sql_table('Table2', engine)

    # define features and label arrays
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """ Function to normalize, tokenize, and lemmatize text 
    Args:
    text: string. ? Message text to be processed tokenized
    
    Returns:
    tokens: list of tokens from text
    """
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #tokenize text
    tokens = word_tokenize(text)
    
    #lemmatize and remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
 
    return tokens


def build_model():
    """
    Function builds ML pipeline 
    
    Output:
        ML Pipeline that processes text and applies a classifier model.
        
    """
    # text processing and model pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()), 
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    # define parameters for GridSearchCV
    parameters = { 'vect__max_df': (0.75, 1.0),
        'tfidf__use_idf': (True, False)}

    # create gridsearch object and return as final model pipeline
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function evaluates ML model, splits data into train/test sets, train pipeline, and prints model performance
  
    
    Arguments:
        pipeline -> ml model pipeline
        X_test -> Test features
        Y_test -> Test labels
        target_names -> label names 
    """
    # Train pipeline, split data into train and test sets
    X_train, X_test, y_train, Y_test = train_test_split(X, Y)
    # train classifier
    model.fit(X_train, y_train)
    # evaluate on test set
    predicted_improved = model.predict(X_test)
    # print results of tuned model
    print(classification_report(Y_test, predicted_improved, category_names=Y_test.columns))
    
                                                  
def save_model(model, model_filepath):
    """
    Save Pipeline function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        pipeline -> GridSearchCV pipeline object
        model_filepath -> destination path to save .pkl file
    
    """

    # save model to disk
    model_filepath = 'final_model.sav'
    # export model as a pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()


