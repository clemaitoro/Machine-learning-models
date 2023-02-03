import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the data
dataset = pd.read_csv("50_Startups.csv")

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# One hot encoding the categorical data (State column)
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [3])], remainder="passthrough")

x = np.array(ct.fit_transform(x))


# Split into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# Training Multiple Linear regression

regression = LinearRegression()

regression.fit(x_train, y_train)

# Predicting the test results

y_pred = regression.predict(x_test)

# Printing the results of the prediction and comparing them to the test set

np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))


