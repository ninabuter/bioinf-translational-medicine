""""
Naive Bayes Classifier
"""
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Load the dataset
df = pd.read_csv("Feature_Selection_PCA", sep="\t")

# Assuming 'Sample' and 'Subgroup' are the identifiers and labels respectively
X = df.drop(['Sample', 'Subgroup'], axis=1)
y = df['Subgroup']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# Initialize Gaussian Naive Bayes classifier
gnb = GaussianNB()

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
}

# Initialize Grid Search Cross Validation
grid_search = GridSearchCV(estimator=gnb, param_grid=param_grid, cv=5, n_jobs=-1)

# Perform Grid Search to find the best hyperparameters
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Use the best model from Grid Search
best_gnb = grid_search.best_estimator_

# Train the classifier
best_gnb.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = best_gnb.predict(X_test)

# Calculate and print the number of mislabeled points
num_mislabeled = (y_test != y_pred).sum()
total_points = X_test.shape[0]
print(f"Number of mislabeled points out of a total {total_points} points: {num_mislabeled}")

# Calculate and print accuracy scores
accuracy_train = accuracy_score(y_train, best_gnb.predict(X_train))
print('Accuracy score on train dataset:', accuracy_train)

accuracy_test = accuracy_score(y_test, y_pred)
print('Accuracy score on test dataset:', accuracy_test)

# Calculate and print precision scores
precision_train = precision_score(y_train, best_gnb.predict(X_train), average='weighted')
print('Precision score on train dataset:', precision_train)

precision_test = precision_score(y_test, y_pred, average='weighted')
print('Precision score on test dataset:', precision_test)

# Calculate and print  recall scores
recall_train = recall_score(y_train, best_gnb.predict(X_train), average='weighted')
print('Recall score on train dataset:', recall_train)

recall_test = recall_score(y_test, y_pred, average='weighted')
print('Recall score on test dataset:', recall_test)

summary = classification_report(y_test, y_pred)
print('summary:', summary)

