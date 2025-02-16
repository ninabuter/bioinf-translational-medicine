""""
Double cross-validation with GNB, kNN, RF, and SVM on the train set.
"""

import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Select the RFE, PCA, and evaluation dataset directory
training_dataset_RFE = "Feature_Selection_RFE_F_SelectKBest"
testing_dataset = "df_call_eval"

# Load the dataset
df = pd.read_csv(training_dataset_RFE, sep="\t")

# Define X and y
X = df.drop(['Sample', 'Subgroup'], axis=1)
y = df['Subgroup']

# Define classifiers and their respective parameter grids for hyperparameter tuning
classifiers = {
    'GNB': (GaussianNB(), {'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7]}),
    'kNN': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 9]}),
    'RandomForest': (RandomForestClassifier(),
                     {'n_estimators': [50, 100, 150, 200],
                      'max_depth': [2, 4, 6, 8],
                      'min_samples_split': [2, 4, 6, 8]}),
    'SVM': (SVC(), {'C': [0.01, 0.1, 1, 10], 'gamma': [50, 60, 70, 80],
                    'kernel': ['linear', 'rbf']})
}

# Define the outer and inner cross-validation folds
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

# Perform double cross-validation for each classifier
for name, (classifier, param_grid) in classifiers.items():
    print(f"Classifier: {name}")
    grid_search = GridSearchCV(classifier, param_grid, cv=inner_cv, scoring='accuracy')
    scores = cross_val_score(grid_search, X, y, cv=outer_cv)
    print(f"Mean accuracy: {scores.mean():.3f}, Std deviation: {scores.std():.3f}")


# Code for predicting subgroups of the evaluation dataset

test_df = pd.read_csv(testing_dataset, sep="\t")
test_samples = test_df['Sample']
test_data = test_df.drop(['Sample'], axis=1)
predicted_subgroups = grid_search.best_estimator_.predict(test_data)  # Best model (SVM) used to make the predictions
predictions_df = pd.DataFrame({'Sample': test_samples, 'Subgroup': predicted_subgroups})
predictions_df.to_csv("predictions.txt", sep="\t", index=False)


# Other

# # Load the test set
# test_df = pd.read_csv(testing_dataset, sep="\t")
#
# # Assuming the test set has the same features as the training set (excluding identifiers)
# X_test = test_df.drop(['Sample', 'Subgroup'], axis=1)
# y_test = test_df['Subgroup']
#
# # Preprocess the test set (if necessary)
# # For example, apply the same transformations (scaling, encoding) that were applied to the training set.
#
# # Evaluate each model on the test set
# for name, (classifier, _) in classifiers.items():  # Assuming 'classifiers' is defined as in the previous code
#     print(f"Evaluating {name} on the test set...")
#
#     # Assuming the model is already trained (from the double cross-validation)
#     # Use the 'predict' method to generate predictions for the test set
#     y_pred = classifier.predict(X_test)
#
#     # Calculate performance metrics
#     accuracy = accuracy_score(y_test, y_pred)
#     # Calculate other metrics as needed (e.g., precision, recall, F1-score)
#
#     print(f"Accuracy: {accuracy:.3f}")
#     # Print other performance metrics
