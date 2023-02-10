import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------- Building linear regression model ---------------------
dataset = pd.read_csv("Position_Salaries.csv")

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(x, y)

# --------------------- Building the polynomial model ------------------

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)

x_poly = poly_reg.fit_transform(x)

lin_reg_2 = LinearRegression()

lin_reg_2.fit(x_poly, y)

# --------------- Visulising results -----------------------------------

plt.scatter(x, y, color="red")
plt.plot(x, lin_reg.predict(x), color="blue")

plt.title("Linear regression")
plt.xlabel("job level")
plt.ylabel("Money")

plt.show()


# Polynomial results
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(x, y, color="red")
plt.plot(x, lin_reg_2.predict(x_poly), color="blue")

plt.title("Polynomial regression")
plt.xlabel("job level")
plt.ylabel("Money")

plt.show()

# Predicting one result for linear and Poly

print(lin_reg.predict([[6.5]])) # Linear

print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))) #Poly

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))

