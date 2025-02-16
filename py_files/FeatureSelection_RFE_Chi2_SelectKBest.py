""""
This script performs feature selection using SelectKBest with Chi2 on a given dataset.
It selects the top 10 features based on their importance in predicting the target variable.
The selected features are then used to create a new dataset, which is saved to a CSV file.
"""

import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2


# Function to load the dataset from a CSV file and preprocess it
def load_data(filepath):
    df = pd.read_csv(filepath, sep="\t")
    df[df == -1] = 9999  # Replace -1 values with 9999 because the chi2 test requires non-negative input features
    return df


# Function to perform feature selection using SelectKBest with Chi-squared test
def feature_selection(df, n_features):
    X = df.drop(['Sample', 'Subgroup'], axis=1)  # Extract features
    y = df['Subgroup']  # Extract target variable
    selector = SelectKBest(chi2, k=n_features)  # Initialize SelectKBest with chi2
    X_new = selector.fit_transform(X, y)  # Fit SelectKBest and transform X
    selected_features = pd.DataFrame(selector.inverse_transform(X_new),
                                     index=df.index,
                                     columns=X.columns)  # Inverse transform to get selected features
    selected_columns = selected_features.columns[selected_features.var() != 0]  # Filter out features with zero variance
    # Get scores of selected features
    feature_scores = sorted(zip(X.columns, selector.scores_), key=lambda x: x[1], reverse=True)
    return selected_columns, feature_scores


# Main function
def main():
    # Load the dataset
    df = load_data("df_call_transposed")

    # Perform feature selection to select top 10 features
    selected_columns, feature_scores = feature_selection(df, 10)

    # Print the selected features in descending order of importance
    print("Top 10 selected features:")
    for i, (feature, score) in enumerate(feature_scores[:10], 1):
        print(f"{i}. {feature} - Score: {score}")

    # Create the final dataset with selected features and save it to a CSV file
    selected_df = df[['Sample', 'Subgroup'] + list(selected_columns)]  # Include 'Sample' and 'Subgroup' columns

    # Replace 9999 with -1
    selected_df.replace(9999, -1, inplace=True)

    # Save it to a CSV file
    selected_df.to_csv("Feature_Selection_RFE_Chi2_SelectKBest", sep='\t', index=False)


# Execute the main function if the script is run directly
if __name__ == "__main__":
    main()


