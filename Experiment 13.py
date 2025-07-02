import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, accuracy_score
)

# Load dataset
data = pd.read_csv("C:\Users\DEVESH\Documents\Dataset\New folder")

# Drop unnecessary columns
data.drop(["CarName"], axis=1, inplace=True)

# Convert categorical columns to numeric
data = pd.get_dummies(data, drop_first=True)

# Visualize correlation matrix
numeric_data = data.select_dtypes(include=[np.number])
plt.figure(figsize=(20, 15))
sns.heatmap(numeric_data.corr(), cmap="coolwarm", annot=True)
plt.title("Feature Correlation Heatmap")
plt.show()

# ----------------------------
# Regression Models
# ----------------------------

X = data.drop(columns=["price"])
y = data["price"]

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(xtrain, ytrain)
dt_preds = dt_model.predict(xtest)

print("\nDecision Tree Regressor:")
print("MAE:", mean_absolute_error(ytest, dt_preds))
print("MSE:", mean_squared_error(ytest, dt_preds))
print("RMSE:", np.sqrt(mean_squared_error(ytest, dt_preds)))
print("R^2 Score:", r2_score(ytest, dt_preds))

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(xtrain, ytrain)
rf_preds = rf_model.predict(xtest)

print("\nRandom Forest Regressor:")
print("MAE:", mean_absolute_error(ytest, rf_preds))
print("MSE:", mean_squared_error(ytest, rf_preds))
print("RMSE:", np.sqrt(mean_squared_error(ytest, rf_preds)))
print("R^2 Score:", r2_score(ytest, rf_preds))

# ----------------------------
# Classification from Regression
# ----------------------------

# Create binary target
median_price = data["price"].median()
data["price_label"] = np.where(data["price"] > median_price, 1, 0)

X_cls = data.drop(columns=["price", "price_label"])
y_cls = data["price_label"]

xtrain_cls, xtest_cls, ytrain_cls, ytest_cls = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

# Decision Tree Classifier
clf_model = DecisionTreeClassifier(random_state=42)
clf_model.fit(xtrain_cls, ytrain_cls)
cls_preds = clf_model.predict(xtest_cls)

print("\nClassification (High/Low Price):")
print("Accuracy:", accuracy_score(ytest_cls, cls_preds))

# ----------------------------
# Summary of Mean Errors
# ----------------------------

print("\nSummary of Mean Errors:")
print("Random Forest MAE:", mean_absolute_error(ytest, rf_preds))
print("Random Forest MSE:", mean_squared_error(ytest, rf_preds))
print("Random Forest RMSE:", np.sqrt(mean_squared_error(ytest, rf_preds)))
