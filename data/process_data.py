# import packages
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Function loads messages and categories data files and creates a new dataset from both files combined
    
    Arguments:
        messages_filepath -> Path to the CSV file containing messages
        categories_filepath -> Path to the CSV file containing categories
    Output:
        df -> database containing messages and categories data combined
    """
    # read in files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # create combined dataset from both files
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """
    Clean Categories Data Function
    
    Arguments:
        df -> Combined dataset containing messages and categories
    Outputs:
        df -> Combined dataset containing messages and categories with categories cleaned up and converted into binary data for machine learning model use
    """
    # split categories into separate category columns, create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';',expand=True)
    # extract a list of new column names for categories
    row = categories.iloc[0]
    category_colnames = list(row.apply(lambda x: x.split("-")[-2]))
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # convert category values to binary values
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.get(-1)
        # convert column from string to numeric
        categories[column] =  pd.to_numeric(categories[column])
    
    # replace category column in df with new category columns
    # drop the original categories column from `df`
    df.drop(columns='categories', inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    
    # remove duplicates
    df.drop_duplicates(inplace=True)
    
    # convert rows where category related = 2 to instead be related = 1, in order to only have binary category values
    df.loc[(df.related == 2),'related']=1
    
    return df


def save_data(df, database_filename):
    """
    Functions to save data to SQLite database 
    
    Arguments:
        df -> Cleaned dataset containing combination of messages and categories datasets
    """
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Table2', engine, index=False,if_exists='replace')
    

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

