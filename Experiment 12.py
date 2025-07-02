import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
iris = pd.read_csv("C:\Users\DEVESH\Documents\Dataset")
print(iris.head()) 
print()
print(iris.describe())
print("Target Labels:", iris["species"].unique())

# Plot using plotly
import plotly.express as px
fig = px.scatter(iris, x="sepal_width", y="sepal_length", color="species")
fig.show()

# Features and target
x = iris.drop("species", axis=1)
y = iris["species"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# -------------------- KNN Model --------------------
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)

# -------------------- Naive Bayes Model --------------------
nb = GaussianNB()
nb.fit(x_train, y_train)

# Predict for new input
x_new = np.array([[6, 2.9, 1, 0.2]])
knn_pred = knn.predict(x_new)
nb_pred = nb.predict(x_new)

print("\nPrediction for New Input [6, 2.9, 1, 0.2]:")
print("KNN Prediction:", knn_pred[0])
print("Naive Bayes Prediction:", nb_pred[0])

# -------------------- Evaluation on Test Set --------------------
y_pred_knn = knn.predict(x_test)
y_pred_nb = nb.predict(x_test)

print("\nKNN Classification Report:")
print(classification_report(y_test, y_pred_knn))

print("Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb))

print("\nKNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
