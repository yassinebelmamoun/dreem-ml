from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np


def trainer(df):
    model = LinearRegression()
    
    # Train the model 
    X = df.drop(['power_increase'], axis=1)
    y = df['power_increase']
    model.fit(X, y)
    # End of training

    return model


