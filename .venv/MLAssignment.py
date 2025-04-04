import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_data():
    """Load and return the car evaluation dataset"""
    car_evaluation = fetch_ucirepo(id=19)
    X = car_evaluation.data.features
    y = car_evaluation.data.targets
    print("\nOriginal data sample:")
    print(X.head())
    return X, y


def encode_features(X):
    """One-hot encode features without dropping categories"""
    encoder = OneHotEncoder(sparse_output=False)  # Keep all categories
    X_encoded = encoder.fit_transform(X)
    print(f"\nEncoded shape: {X_encoded.shape}")
    return X_encoded, encoder


def compute_similarity(X_encoded):
    """Compute and return cosine similarity matrix"""
    sim_matrix = cosine_similarity(X_encoded)
    print("\nSimilarity matrix sample (5x5):")
    print(pd.DataFrame(sim_matrix[:5, :5]).round(2))
    return sim_matrix


def perform_clustering(X_encoded, sim_matrix, original_df):
    """Perform and visualize clustering using both features and similarity"""
    # Convert similarity to distance
    distance_matrix = 1 - sim_matrix

    # Determine optimal k using elbow method
    distortions = []
    K_range = range(2, 8)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_encoded)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(10, 5))
    plt.plot(K_range, distortions, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal k')
    plt.savefig('visualizations/elbow_method.png', bbox_inches='tight')
    plt.close()

    # Cluster using KMeans on feature space
    optimal_k = 4  # Based on elbow method
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    feature_clusters = kmeans.fit_predict(X_encoded)

    # Cluster using Agglomerative on distance matrix
    agg = AgglomerativeClustering(n_clusters=optimal_k,
                                  metric='precomputed',  # Changed from affinity
                                  linkage='average')
    similarity_clusters = agg.fit_predict(distance_matrix)

    # Add clusters to original dataframe
    original_df['Feature_Cluster'] = feature_clusters
    original_df['Similarity_Cluster'] = similarity_clusters

    # Dimensionality reduction for visualization
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    pos = mds.fit_transform(distance_matrix)

    # Visualization
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x=pos[:, 0], y=pos[:, 1], hue=feature_clusters,
                    palette='viridis', s=100, alpha=0.7)
    plt.title('KMeans Clustering on Feature Space')

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=pos[:, 0], y=pos[:, 1], hue=similarity_clusters,
                    palette='viridis', s=100, alpha=0.7)
    plt.title('Agglomerative Clustering on Distance Matrix')

    plt.tight_layout()
    plt.savefig('visualizations/clustering_results.png', bbox_inches='tight')
    plt.close()

    return original_df


def analyze_and_visualize(sim_matrix, original_df, y):
    """Analyze similarity pairs and create visualizations"""
    os.makedirs('visualizations', exist_ok=True)

    masked_matrix = sim_matrix.copy()
    np.fill_diagonal(masked_matrix, np.nan)

    max_sim = np.nanmax(masked_matrix)
    min_sim = np.nanmin(masked_matrix)

    print(f"\nMaximum similarity: {max_sim:.2f}")
    print(f"Minimum similarity: {min_sim:.2f}")

    # Extreme pairs analysis
    max_indices = np.where((masked_matrix == max_sim) & (~np.isnan(masked_matrix)))
    min_indices = np.where((masked_matrix == min_sim) & (~np.isnan(masked_matrix)))

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(sim_matrix[:50, :50], ax=axes[0], cmap='YlOrRd', vmin=0, vmax=1)
    axes[0].set_title("Similarity Matrix (First 50 Cars)")

    # Cluster distribution plot
    cluster_counts = original_df['Feature_Cluster'].value_counts()
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=axes[1], palette='viridis')
    axes[1].set_title("Cluster Distribution")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig('visualizations/similarity_and_clusters.png', bbox_inches='tight')
    plt.close()


def main():
    # Load and preprocess data
    X, y = load_data()
    X_encoded, encoder = encode_features(X)

    # Compute similarity
    sim_matrix = compute_similarity(X_encoded)

    # Perform clustering
    clustered_df = perform_clustering(X_encoded, sim_matrix, X.copy())

    # Analyze and visualize
    analyze_and_visualize(sim_matrix, clustered_df, y)

    # Save clustered data
    clustered_df.to_csv('visualizations/clustered_car_data.csv', index=False)
    print("\nClustered data saved to 'visualizations/clustered_car_data.csv'")
    print("\nAll visualizations saved in 'visualizations' directory")


if __name__ == "__main__":
    main()