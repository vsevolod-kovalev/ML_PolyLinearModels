from itertools import product
from itertools import combinations_with_replacement
import numpy as np
import csv

# This file implements polynomial regression with multiple features
# Features (3): area, n_bedrooms, n_stories
# Target (1): price

def load_data(csv_file_path):
    features = []
    targets = []
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            targets.append(float(row['price']))

            feature_vector = [
                float(row['area']),
                float(row['bedrooms']),
                float(row['stories'])
            ]
            features.append(feature_vector)

    # Convert lists to numpy arrays
    X = np.array(features)
    y = np.array(targets)

    return X, y

# set degree
degree = 2
csv_file_path = 'Housing.csv'
x_train, y_train = load_data(csv_file_path)

# engineer features
print(x_train)

def engineerX(X, degree):
    n_samples, n_features = X.shape
    new_features = []

    # Generate all combinations of features with replacement up to the specified degree
    combinations = list(combinations_with_replacement(range(n_features), degree))

    # Add the polynomial features
    for i in range(n_samples):
        row = []
        for combination in combinations: 
            term = np.prod([X[i, index] for index in combination])
            row.append(term)
        new_features.append(row)

    return np.array(new_features)

# Load data
csv_file_path = 'Housing.csv'
x_train, y_train = load_data(csv_file_path)

# Engineer polynomial features up to degree 2
degree = 2
eng_updated_x = engineerX(x_train, degree)

print("Engineered features:\n", eng_updated_x)

# Function to compute the cost (mean squared error)
def compute_cost(W, b, X, y):
    m = X.shape[0]
    error = np.dot(X, W) + b - y
    cost = np.sum(np.square(error)) / (2 * m)
    return cost


W = np.zeros(eng_updated_x.shape[1])
b = 0
cost = compute_cost(W, b, eng_updated_x, y_train)
print("Initial cost:", cost)

