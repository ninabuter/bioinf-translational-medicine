# manual nested cross-validation for random forest on a classification dataset
from numpy import mean
from numpy import std
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Select the RFE or PCA dataset directory
dataset = "Feature_Selection_RFE_F_SelectKBest"

# Load the dataset
df = pd.read_csv(dataset, sep="\t")

# Assuming 'Sample' and 'Subgroup' are the identifiers and labels respectively
X = df.drop(['Sample', 'Subgroup'], axis=1)
y = df['Subgroup']

# configure the cross-validation procedure
cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
# enumerate splits
outer_results = list()
for train_ix, test_ix in cv_outer.split(X):
    X_train, X_test = X[train_ix, :], X[test_ix, :]  # split data
    y_train, y_test = y[train_ix], y[test_ix]
    cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)  # configure the cross-validation procedure
    model = RandomForestClassifier(random_state=1)  # define the model
    space = dict()  # define search space
    space['n_estimators'] = [10, 100, 500]
    space['max_features'] = [2, 4, 6]
    search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)  # define search
    result = search.fit(X_train, y_train)  # execute search
    best_model = result.best_estimator_  # get the best performing model fit on the whole training set
    yhat = best_model.predict(X_test)  # evaluate model on the hold out dataset
    acc = accuracy_score(y_test, yhat)  # evaluate the model
    outer_results.append(acc)  # store the result
    print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))  # report progress
# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))