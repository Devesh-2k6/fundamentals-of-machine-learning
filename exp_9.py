import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Create some sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5]).reshape(-1, 1)

# Create a linear regression object and fit the data
reg = LinearRegression().fit(X, y)

# Predict new values
X_new = np.array([6]).reshape(-1, 1)
y_pred = reg.predict(X_new)

# Plot the data and the linear regression line
plt.scatter(X, y)
plt.plot(X, reg.predict(X), color='red')
plt.show()

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Create some sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5]).reshape(-1, 1)

# Transform the data to include another axis
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Create a polynomial regression object and fit the data
reg = LinearRegression().fit(X_poly, y)

# Predict new values
X_new = np.array([6]).reshape(-1, 1)
X_new_poly = poly.transform(X_new)
y_pred = reg.predict(X_new_poly)

# Plot the data and the polynomial regression curve
plt.scatter(X, y)
plt.plot(X, reg.predict(X_poly), color='red')
plt.show()

#logistic regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Create a binary classification dataset (linearly separable)
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([0, 0, 0, 1, 1, 1])  # Binary labels

# Create and train logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Predict probabilities across a range of values
X_range = np.linspace(0, 7, 100).reshape(-1, 1)
y_prob = model.predict_proba(X_range)[:, 1]  # Probabilities for class 1

# Plot original points and the logistic curve
plt.scatter(X, y, color='black', label='Original Data')
plt.plot(X_range, y_prob, color='red', label='Logistic Curve')
plt.xlabel('X')
plt.ylabel('Probability of Class 1')
plt.title('Logistic Regression')
plt.legend()
plt.grid(True)
plt.show()

