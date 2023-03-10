# Data Preprocessing Template
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


from sklearn.tree import DecisionTreeRegressor
# ----------------------------- Training the model -----------------
regressor = DecisionTreeRegressor(random_state=0)

regressor.fit(x, y)


# ----------------------------- Predicting results ------------------
y_pred = regressor.predict([[6.5]])

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))


# ------------------------ Visualising results ----------------------
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(x, y, color="red")
plt.plot(x_grid, regressor.predict(x_grid), color="blue")

plt.title("DecissionTree regression")
plt.xlabel("job level")
plt.ylabel("Money")

plt.show()

