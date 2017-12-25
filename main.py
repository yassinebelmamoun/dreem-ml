import settings
import pandas as pd
from pickler import get_or_create_pickle
from train import trainer
from test import predicter

def main():

    file_test, file_train = 'test_all.csv', 'train_all.csv' 
    # file_test, file_train = 'test_1000.csv', 'train_1000.csv' 
    df_test  = get_or_create_pickle(file = file_test)
    df_train = get_or_create_pickle(file = file_train)

    # Train
    model = trainer(df_train)

    # Predict
    predicter(model, df_test)


if __name__ == '__main__':
    main()