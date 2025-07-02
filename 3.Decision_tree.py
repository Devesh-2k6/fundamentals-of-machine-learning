import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split

class DecisionTreeNode:
    def __init__(self, attribute=None, label=None, branches=None):
        self.attribute = attribute
        self.label = label
        self.branches = branches if branches is not None else {}

def entropy(data):
    target = data['target']
    values, counts = np.unique(target, return_counts=True)
    e = 0
    for c in counts:
        p = c / len(target)
        e -= p * math.log2(p)
    return e

def information_gain(data, attribute):
    total = len(data)
    e_attr = 0
    for val in data[attribute].unique():
        subset = data[data[attribute] == val]
        e_attr += (len(subset) / total) * entropy(subset)
    return entropy(data) - e_attr

def id3(data, attributes):
    target = data['target']
    if len(target.unique()) == 1:
        return DecisionTreeNode(label=target.iloc[0])
    if len(attributes) == 0:
        return DecisionTreeNode(label=target.value_counts().idxmax())

    gains = {attr: information_gain(data, attr) for attr in attributes}
    best = max(gains, key=gains.get)
    node = DecisionTreeNode(attribute=best)

    for val in data[best].unique():
        subset = data[data[best] == val].drop(columns=[best])
        if subset.empty:
            node.branches[val] = DecisionTreeNode(label=target.value_counts().idxmax())
        else:
            new_attrs = attributes.copy()
            new_attrs.remove(best)
            node.branches[val] = id3(subset, new_attrs)
    return node

def classify(sample, tree):
    if tree.label is not None:
        return tree.label
    attr = tree.attribute
    val = sample[attr]
    if val in tree.branches:
        return classify(sample, tree.branches[val])
    return None

def print_tree(node, indent=""):
    if node.label is not None:
        print(indent + "Predict:", node.label)
    else:
        for val, subtree in node.branches.items():
            print(indent + f"If {node.attribute} == {val}:")
            print_tree(subtree, indent + "  ")

# Load dataset
data = pd.read_csv(C:\Users\DEVESH\Documents\Dataset")

# Split into train and test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
attributes = train_data.columns[:-1].tolist()

# Train the tree
root = id3(train_data, attributes)

# Print the tree
print("\nDecision Tree:")
print_tree(root)

# Test the tree
print("\nTesting on Test Data:")
correct = 0
for idx, row in test_data.iterrows():
    predicted = classify(row, root)
    actual = row['target']
    print(f"Actual: {actual}, Predicted: {predicted}")
    if predicted == actual:
        correct += 1

accuracy = (correct / len(test_data)) * 100
print(f"\nAccuracy on test data: {accuracy:.2f}%")
