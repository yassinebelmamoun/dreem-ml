from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy
import pandas as pd

def predicter(model, df):
    # Predict from model
    if 'power_increase' in df.columns:
        X = df.drop(columns=['power_increase'])
    else:
        X = df
    y_predicted = model.predict(X)

    # Measure error if possible
    if 'power_increase' in df.columns:
        print('MSE: ', mean_squared_error(y_predicted, df['power_increase']))

    # numpy.savetxt("foo.csv", y_predicted, delimiter=",")
    df_predicted = pd.DataFrame(y_predicted)
    df_predicted.to_csv('result/test_1.csv', header=['power_increase'])
