import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


dataset = pd.read_csv("Position_Salaries.csv")

x = dataset.iloc[:, 1:-1].values

y = dataset.iloc[:, -1].values

y = y.reshape(len(y), 1)

# -------------------- Feature scaling ------------------------------
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc2 = StandardScaler()


x = sc.fit_transform(x)
y = sc2.fit_transform(y)

# ----------------------- Training the model -------------------------
from sklearn.svm import SVR

regressor = SVR(kernel="rbf")

regressor.fit(x, y)


# --------- Reversing feature scaling and predicting result ------------------

# print(sc2.inverse_transform(regressor.predict(sc.transform([[6.5]])).reshape(-1,1)))

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))


# -------------------------------- Visualising results ----------------------------------------

# plt.scatter(sc.inverse_transform(x), sc2.inverse_transform(y), color="red")
# plt.plot(sc.inverse_transform(x), sc2.inverse_transform(regressor.predict(x).reshape(-1,1)), color="blue")
#
# plt.title("SVR")
# plt.xlabel("job level")
# plt.ylabel("Money")
#
# plt.show()

# -------------------------------- Visualising in high res --------------
x_grid = np.arange(min(sc.inverse_transform(x)), max(sc.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(sc.inverse_transform(x), sc2.inverse_transform(y), color="red")
plt.plot(x_grid, sc2.inverse_transform(regressor.predict(sc.transform(x_grid)).reshape(-1,1)), color="blue")

plt.title("SVR")
plt.xlabel("job level")
plt.ylabel("Money")

plt.show()
