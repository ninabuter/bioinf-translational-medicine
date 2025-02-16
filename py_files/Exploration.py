""""
This script performs various exploratory data analyses, namely t-SNE 2D/3D visualizations.
"""

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

# Import data
df_call_transposed = pd.read_csv("df_call_transposed", sep="\t")

# T-SNE
numeric_data = df_call_transposed.drop(['Sample', 'Subgroup'], axis=1)  # Only numerical data for T-SNE
X = numeric_data.values  # Convert df to numpy array
tsne = TSNE(n_components=3, perplexity=5, random_state=42)  # Initiate the TSNE tool
X_embedded = tsne.fit_transform(X)  # Perform t-SNE
subgroups = df_call_transposed['Subgroup']  # Extract subgroup labels

# Visualize the high-dimensional data with T-SNE tool (not color-coded)
# embedded_df = pd.DataFrame(X_embedded, columns=['Dimension 1', 'Dimension 2'])  # Create df for embedded data
# plt.figure(figsize=(10, 8))  # Plot the t-SNE data
# plt.scatter(embedded_df['Dimension 1'], embedded_df['Dimension 2'], alpha=0.5)
# plt.title('t-SNE Visualization')
# plt.xlabel('Dimension 1')
# plt.ylabel('Dimension 2')
# plt.show()

# Same visualization, but with subgroup color-coding
plt.figure(figsize=(10, 8))  # Plot the embedded data, color-coded by subgroup
for subgroup in subgroups.unique():  # Plot each subgroup separately, using different color for each subgroup
    mask = (subgroups == subgroup)
    plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1], label=subgroup, alpha=0.7)
plt.title('t-SNE Visualization (Color-coded by Subgroup)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
# Save the figure
plt.savefig('tSNE_2D.png')
plt.show()

# 3D visualization
# fig = plt.figure(figsize=(10, 8))  # Plot the embedded data in 3D
# ax = fig.add_subplot(111, projection='3d')
# for subgroup in subgroups.unique():  # Plot each subgroup separately
#     mask = (subgroups == subgroup)
#     ax.scatter(X_embedded[mask, 0], X_embedded[mask, 1], X_embedded[mask, 2], label=subgroup, alpha=0.7)
# ax.set_title('t-SNE Visualization in 3D')
# ax.set_xlabel('Dimension 1')
# ax.set_ylabel('Dimension 2')
# ax.set_zlabel('Dimension 3')
# ax.legend()
# plt.show()

# K-means clustering on the embedded data from T-SNE
# kmeans = KMeans(n_clusters=3, random_state=42)
# clusters = kmeans.fit_predict(X_embedded)
# plt.figure(figsize=(10, 8))  # Plot the embedded data, color-coded by cluster
# for cluster_id in np.unique(clusters):
#     mask = (clusters == cluster_id)
#     plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1], label=f'Cluster {cluster_id}', alpha=0.7)
# plt.title('t-SNE Visualization with K-means Clustering')
# plt.xlabel('Dimension 1')
# plt.ylabel('Dimension 2')
# plt.legend()
# plt.show()
# cluster_labels = {} # Obtain the labels of the samples that are clustered together in the embedded data
# for cluster_id in np.unique(clusters):
#     mask = (clusters == cluster_id)
#     cluster_samples = df_call_transposed[mask]['Sample'].tolist()
#     cluster_labels[cluster_id] = cluster_samples
# for cluster_id, samples in cluster_labels.items():
#     print(f"Cluster {cluster_id}: {samples}")

# Neighborhood relationships
# distances = euclidean_distances(X_embedded)  # Calculate pairwise distances between points in the embedded space
# plt.figure(figsize=(10, 8))  # Plot the distances as a heatmap
# plt.imshow(distances, cmap='viridis', origin='lower')
# plt.colorbar(label='Distance')
# plt.title('Pairwise Distance Heatmap')
# plt.xlabel('Data Point Index')
# plt.ylabel('Data Point Index')
# plt.show()