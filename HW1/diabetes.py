#####################################################################
# Omar Abdelmotaleb
# Assignment 1
# CS-559-B
# I pledge my honor that I have abided by the Stevens Honor System.
####################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df      = pd.read_csv("diabetes.csv")
glucose = df["Glucose"].to_list()
bmi     = df["BMI"].to_list()
outcome = df["Outcome"].to_list()

## Logistic Regression Model
# Adapted from a tutorial
# https://www.youtube.com/watch?v=JDU3AzH3WKg&ab_channel=PythonEngineer 

class LogisticRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate # Learning Rate
        self.n_iters = n_iters  # Num. of iterations for descent
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

output          = list (map (  lambda x,y: [x,y], glucose,bmi    ))
output_array    = np.array(output)
outcome_array   = np.array(outcome)

X, y = output_array, outcome_array

# Used sklearn to just split test and training data.
# Following previous tutorial for values.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

regressor = LogisticRegression(learning_rate=0.0001, n_iters=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

print("LR classification accuracy results:", accuracy(y_test, predictions))


######################

bp  = df["BloodPressure"].to_list()
st  = df["SkinThickness"].to_list()
age = df["Age"].to_list()
ins = df["Insulin"].to_list()
dpf = df["DiabetesPedigreeFunction"].to_list()

output2          = list (map (  lambda x,y,u,w,k: [x,y,u,w,k], bp,st,age,ins,dpf    ))
output_array    = np.array(output2)
outcome_array   = np.array(outcome)

X, y = output_array, outcome_array

# Used sklearn to just split test and training data.
# Following previous tutorial for values.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

regressor = LogisticRegression(learning_rate=0.0001, n_iters=10000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

print("LR 2 classification accuracy results:", accuracy(y_test, predictions))


