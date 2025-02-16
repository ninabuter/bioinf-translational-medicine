""""
This script performs feature selection using SelectKBest with ANOVA F-value (f_classif) 
on a given dataset. It selects the top 10 features based on their importance in predicting 
the target variable. The selected features are then used to create a new dataset, which is 
saved to a CSV file.
"""

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

# Function to load the dataset from a CSV file
def load_data(filepath):
    df = pd.read_csv(filepath, sep="\t")
    return df


# Function to perform feature selection using SelectKBest
def feature_selection(df, n_features):
    X = df.drop(['Sample', 'Subgroup'], axis=1)  # Extract features
    y = df['Subgroup']  # Extract target variable
    selector = SelectKBest(f_classif, k=n_features)  # Initialize SelectKBest
    X_new = selector.fit_transform(X, y)  # Fit SelectKBest and transform X
    selected_features = pd.DataFrame(selector.inverse_transform(X_new),
                                     index=df.index,
                                     columns=X.columns)  # Inverse transform to get selected features
    selected_columns = selected_features.columns[selected_features.var() != 0]  # Filter out features with zero variance

    # Get scores of selected features
    feature_scores = sorted(zip(X.columns, selector.scores_), key=lambda x: x[1], reverse=True)

    return selected_columns, feature_scores


# Function to create a new dataset with selected features
def create_selected_dataset(df, selected_columns):
    selected_df = df[['Sample', 'Subgroup'] + list(selected_columns)]  # Include 'set' and 'solubility' columns
    return selected_df


# Function to plot the top ten features
def plot_top_features(feature_scores):
    features, scores = zip(*feature_scores)

    # Truncate feature names to a certain length
    max_length = 15  # Maximum length of feature names
    truncated_features = [feature[:max_length] + '...' if len(feature) > max_length else feature for feature in
                          features]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(truncated_features)), scores, align='center', color='skyblue')
    plt.yticks(range(len(truncated_features)), truncated_features)
    plt.xlabel('Feature Score')
    plt.title('Top 10 Selected Features')
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest score at the top
    plt.show()


# Main function
def main():
    # Load the dataset
    df = load_data("df_call_transposed")

    # Get all features except 'set' and 'solubility'
    all_features = set(df.columns) - {'Sample', 'Subgroup'}

    # Perform feature selection to select top 10 features
    selected_columns, feature_scores = feature_selection(df, 10)

    # Print the selected features in descending order of importance
    print("Top 10 selected features:")
    for i, (feature, score) in enumerate(feature_scores[:10], 1):
        print(f"{i}. {feature} - Score: {score}")

    # Calculate removed features
    removed_features = all_features - set(selected_columns)
    print(f"\nRemoved features: {removed_features}")
    print(f"Number of features left: {len(selected_columns)}")

    # Plot the top ten features
    plot_top_features(feature_scores[:10])

    # Create the final dataset with selected features and save it to a CSV file
    selected_df = create_selected_dataset(df, selected_columns)
    selected_df.to_csv("Feature_Selection_RFE_F_SelectKBest", sep='\t', index=False)


# Execute the main function if the script is run directly
if __name__ == "__main__":
    main()

