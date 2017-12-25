from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy
import pandas as pd

def predicter(model, df):
    X_cols = list(df.columns.values)
    try:
        X_cols.remove('power_increase')
    except:
        print('This is not a training set!')
        pass

    # Predict from model
    X = df[X_cols]
    y_predicted = model.predict(X)

    # Measure error if possible
    try:
        y_true = df[['power_increase']].as_matrix()
        print('MSE: ', mean_squared_error(y_predicted, y_true))
    except:
        print('Unpredictable')

    # numpy.savetxt("foo.csv", y_predicted, delimiter=",")
    df_predicted = pd.DataFrame(y_predicted)
    df_predicted.to_csv('result/test_1.csv', header=['power_increase'])
