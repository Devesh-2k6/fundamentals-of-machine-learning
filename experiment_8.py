# Step 1: Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Step 2: Generate synthetic dataset
X, y = make_classification(
    n_samples=100,
    n_features=1,
    n_classes=2,
    n_clusters_per_class=1,
    flip_y=0.03,
    n_informative=1,
    n_redundant=0,
    random_state=42
)

# Step 3: Visualize the data
plt.figure(figsize=(6, 4))
plt.scatter(X, y, c=y, cmap='rainbow', edgecolor='k', s=80)
plt.title('Scatter Plot of Classification Data')
plt.xlabel('Feature Value')
plt.ylabel('Class')
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Step 5: Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Predict and get confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Step 7: Plot confusion matrix correctly
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues', ax=ax)
plt.title('Confusion Matrix - Logistic Regression')
plt.show()
