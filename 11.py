import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB

# Load the dataset
data = pd.read_csv(r"C:\Users\sri\Desktop\New folder\CREDITSCORE.csv")
print(data.head())
print(data.info())

# Encode 'Credit_Mix'
data["Credit_Mix"] = data["Credit_Mix"].map({
    "Bad": 0,
    "Standard": 1,
    "Good": 2
})

# Make sure there are no missing mappings
print("\nUnique values in Credit_Mix after encoding:")
print(data["Credit_Mix"].unique())

# Define features and label
x = np.array(data[["Annual_Income", "Monthly_Inhand_Salary",
                   "Num_Bank_Accounts", "Num_Credit_Card",
                   "Interest_Rate", "Num_of_Loan",
                   "Delay_from_due_date", "Num_of_Delayed_Payment",
                   "Credit_Mix", "Outstanding_Debt",
                   "Credit_History_Age", "Monthly_Balance"]])

y = np.array(data["Credit_Score"])

# Split the dataset
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.33, random_state=42)

# Train the Random Forest Classifier
model1 = GaussianNB()
model1.fit(xtrain, ytrain)
model = RandomForestClassifier()
model.fit(xtrain, ytrain)

# Predict using the test data
y_pred1 = model1.predict(xtest)
y_pred = model.predict(xtest)

# Print predictions
print("Predicted Credit Scores on RMS Test Data:")
print(y_pred)
print("Predicted Credit Scores on NVB Test Data:")
print(y_pred1)

# Evaluate the model
print("\nClassification RMS Report:")
print(classification_report(ytest, y_pred))

print("\nConfusion RMS Matrix:")
print(confusion_matrix(ytest, y_pred))

print("\nClassification NVB Report:")
print(classification_report(ytest, y_pred1))

print("\nConfusion NVB Matrix:")
print(confusion_matrix(ytest, y_pred1))

#plot

import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Encode labels if they are not numeric (optional safeguard)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ytest_encoded = le.fit_transform(ytest)
y_pred_encoded = le.transform(y_pred)
y_pred1_encoded = le.transform(y_pred1)

# Calculate metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
rf_scores = [
    accuracy_score(ytest_encoded, y_pred_encoded),
    precision_score(ytest_encoded, y_pred_encoded, average='weighted'),
    recall_score(ytest_encoded, y_pred_encoded, average='weighted'),
    f1_score(ytest_encoded, y_pred_encoded, average='weighted')
]

nb_scores = [
    accuracy_score(ytest_encoded, y_pred1_encoded),
    precision_score(ytest_encoded, y_pred1_encoded, average='weighted'),
    recall_score(ytest_encoded, y_pred1_encoded, average='weighted'),
    f1_score(ytest_encoded, y_pred1_encoded, average='weighted')
]

# Plotting
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, rf_scores, width, label='Random Forest', color='royalblue')
plt.bar(x + width/2, nb_scores, width, label='Naive Bayes', color='mediumseagreen')

plt.xlabel('Evaluation Metrics')
plt.ylabel('Score')
plt.title('Comparison of Random Forest and Naive Bayes')
plt.xticks(x, metrics)
plt.ylim(0, 1.1)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Optional: Encode labels if they are strings
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ytest_encoded = le.fit_transform(ytest)
y_pred_encoded = le.transform(y_pred)
y_pred1_encoded = le.transform(y_pred1)

# Compute confusion matrices
cm_rf = confusion_matrix(ytest_encoded, y_pred_encoded)
cm_nb = confusion_matrix(ytest_encoded, y_pred1_encoded)

# Define class labels
class_names = le.classes_

# Plot side-by-side confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Random Forest
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=class_names, yticklabels=class_names)
axes[0].set_title("Random Forest Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# Naive Bayes
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=class_names, yticklabels=class_names)
axes[1].set_title("Naive Bayes Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()
