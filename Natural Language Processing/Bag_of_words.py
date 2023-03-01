import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

# --------------------------- Cleaning the text ------------------------

import re
import nltk

nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0, 1000):
    review = re.sub("[^a-zA-Z]", " ", dataset["Review"][i])
    review = review.lower()
    review = review.split()

    ps = PorterStemmer()
    all_stopwords = stopwords.words("english")
    all_stopwords.remove("not")
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = " ".join(review)

    corpus.append(review)


# ----------- Creating BoW model -------------------------------------

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# -------------- Splitting between train and test set -----------------
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# ------------------- Training the model -----------------------------
from sklearn.naive_bayes import GaussianNB

regressor = GaussianNB()

regressor.fit(x_train, y_train)

# -------------------------------- Predicting results ---------------


y_pred = regressor.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))


# --------------------------- Predicting accuracy ---------------------
from sklearn.metrics import accuracy_score,confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)

print(matrix)
print(accuracy)

