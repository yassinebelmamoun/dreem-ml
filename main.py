import pandas as pd
from sklearn.externals import joblib

from pickler import get_or_create_pickle
from train import trainer
from test import predicter
from utils import data_processer

def get_initial_dataset(file):    
    # Render dataframe from csv file
        # IF pickle does exist:      render dataframe 
        # ELSE                :      create pickle from csv and render dataframe
    print('Generate the dataframe...')
    df = get_or_create_pickle(file)
    return df 

def step_processing(df, name):
    # Processing data (function: data_processer in utils.py)  
    # Steps: Normalize
    print('Processing Data...')
    df = data_processer(df)
    print('Saving to Dataframe..')
    df.to_pickle('dataset/' + name)
    print('Dataframe created:\t\t', name)

def get_processed_data(name):
    print('Get processed dataframe...')
    return pd.read_pickle('dataset/' + name)

def step_training(model_name, df_train):
    # Train
    print("Training Model..")
    model = trainer(df_train)
    print("Saving Model..")
    joblib.dump(model, 'model/' + model_name)

def get_model(model_name):
    return joblib.load('model/' + model_name)

def step_predicting(model, df_test):
    # Predict
    print('Predicting..')
    predicter(model, df_test)

if __name__ == '__main__':
    main()