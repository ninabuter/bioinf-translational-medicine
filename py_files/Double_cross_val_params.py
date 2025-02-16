"""""
Nested cross-validation and prediction on the evaluation set.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Select the RFE, PCA, and evaluation dataset directory
training_dataset_RFE = "Feature_Selection_RFE_F_SelectKBest"
training_dataset_PCA = "Feature_Selection_PCA"
testing_dataset_RFE = "df_call_eval_RFE"

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

# Outer and inner cross-validation folds
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

# Nested cross-validation for each classifier
for name, (classifier, param_grid) in classifiers.items():
    print(f"Classifier: {name}")
    outer_results = []
    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        grid_search = GridSearchCV(classifier, param_grid, cv=inner_cv, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        # Print best hyperparameters
        print(f"Best hyperparameters for {name}: {grid_search.best_params_}")

        # Evaluate on the test fold
        score = grid_search.score(X_test, y_test)
        outer_results.append(score)

    mean_score = sum(outer_results) / len(outer_results)
    std_dev = np.std(outer_results)
    print(f"Mean accuracy: {mean_score:.3f}, Std deviation: {std_dev:.3f}")
    print(grid_search.best_params_)

# Code for predicting subgroups of the evaluation dataset
test_df = pd.read_csv(testing_dataset_RFE, sep="\t")
test_samples = test_df['Sample']
test_data = test_df.drop(['Sample'], axis=1)
predicted_subgroups = grid_search.best_estimator_.predict(test_data)  # Best model (SVM) used to make the predictions
predictions_df = pd.DataFrame({'Sample': test_samples, 'Subgroup': predicted_subgroups})
with open("prediction.txt", "w") as file:
    file.write('"Sample"\t"Subgroup"\n')
    for index, row in predictions_df.iterrows():
        file.write(f'"{row["Sample"]}"\t"{row["Subgroup"]}"\n')

