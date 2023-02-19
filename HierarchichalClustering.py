import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch


# -------------------- Importing the data ------------------------------

dataset = pd.read_csv("Mall_Customers.csv")

x = dataset.iloc[:, [3, 4]].values


# ------- Using dendograms to find the optimal number of clusters ------
# dendrogram = sch.dendrogram(sch.linkage(x, method="ward"))
#
# plt.title("Dendrogram")
# plt.xlabel("Customers")
# plt.ylabel("Euclidean distances")
# plt.show()


# ----------------------- Training the model ---------------------------
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=5, metric="euclidean", linkage="ward")
y_pred = hc.fit_predict(x)


# ---------------------- Visualising Resutls --------------------------

plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], s=100, c="red", label="cluster1")
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], s=100, c="blue", label="cluster2")
plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1], s=100, c="green", label="cluster3")
plt.scatter(x[y_pred == 3, 0], x[y_pred == 3, 1], s=100, c="cyan", label="cluster4")
plt.scatter(x[y_pred == 4, 0], x[y_pred == 4, 1], s=100, c="magenta", label="cluster5")



plt.title("Clusters of customers")
plt.xlabel("Annual income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()