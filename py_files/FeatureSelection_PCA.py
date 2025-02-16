""""
This script performs feature selection using PCA on a given dataset. It selects the top 10 features based on their
importance in predicting the target variable. The selected features are then used to create a new dataset, which is
saved to a CSV file.
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Function to perform feature selection using PCA
def feature_selection_pca(df, n_components):
    X = df.drop(['Sample', 'Subgroup'], axis=1)  # Extract features
    y = df['Subgroup']  # Extract target variable
    scaler = StandardScaler()  # Initialize StandardScaler
    X_scaled = scaler.fit_transform(X)  # Standardize the features
    pca = PCA(n_components=n_components)  # Initialize PCA
    X_pca = pca.fit_transform(X_scaled)  # Fit PCA and transform X
    explained_variance_ratio = pca.explained_variance_ratio_
    return X_pca, explained_variance_ratio


# Main function
def main():
    # Load the dataset
    df = pd.read_csv("df_call_transposed", sep="\t")

    # Perform feature selection using PCA to select top 10 components
    X_pca, explained_variance_ratio = feature_selection_pca(df, 10)

    # Print explained variance ratio of selected components
    print("Explained Variance Ratio of Selected Components:")
    print(explained_variance_ratio)

    # Create DataFrame with selected components
    selected_df = pd.DataFrame(X_pca, columns=[f"PC{i}" for i in range(1, 11)])

    # Add 'Sample' and 'Subgroup' columns to the selected DataFrame
    selected_df = pd.concat([df[['Sample', 'Subgroup']], selected_df], axis=1)

    # Save the selected DataFrame to a CSV file
    selected_df.to_csv("Feature_Selection_PCA", sep='\t', index=False)


# Execute the main function if the script is run directly
if __name__ == "__main__":
    main()
