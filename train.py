from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np


def trainer(df):
    model = LinearRegression()
    # model = SVR(kernel='poly', C=1e3, degree=2)
    # Get name of columns for features and target
    X_cols = list(df.columns.values)
    X_cols.remove('power_increase')

    # Train the model 
    print('Training the model..')
    X = df[X_cols]
    y = df[['power_increase']].as_matrix().ravel() 
    model.fit(X, y)
    print('...End of training')

    return model