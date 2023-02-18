import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# -------------------- Importing the data ------------------------------

dataset = pd.read_csv("Mall_Customers.csv")

x = dataset.iloc[:, [3, 4]].values


# -------------------- Using elbow method + visualisation ------------------------------
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, init="k-means++")
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


plt.plot(range(1, 11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()


# ------------------ Training the model -----------------------------
kmeans = KMeans(n_clusters=5, random_state=42, init="k-means++", n_init="auto")
y_pred = kmeans.fit_predict(x)
print(y_pred)


# ----------------------- Visualising the clusters -------------------------
plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], s=100, c="red", label="cluster1")
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], s=100, c="blue", label="cluster2")
plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1], s=100, c="green", label="cluster3")
plt.scatter(x[y_pred == 3, 0], x[y_pred == 3, 1], s=100, c="cyan", label="cluster4")
plt.scatter(x[y_pred == 4, 0], x[y_pred == 4, 1], s=100, c="magenta", label="cluster5")

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c="yellow", label="Centroids")

plt.title("Clusters of customers")
plt.xlabel("Annual income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()





















