"""
CAR SIMILARITY CLUSTERING MODULE
A comprehensive machine learning pipeline for:
1. Feature encoding and similarity computation
2. Optimal cluster determination
3. Dual clustering (feature space and similarity space)
4. Visualization and analysis
"""

# --------------------------
# IMPORTS
# --------------------------
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import seaborn as sns
import os


# --------------------------
# MAIN CLASS DEFINITION
# --------------------------
class CarSimilarityClusterer(BaseEstimator, ClusterMixin):
    """
    A scikit-learn compatible clustering model for car evaluation data that:
    - Computes pairwise similarities between cars
    - Performs clustering in both feature and similarity spaces
    - Provides comprehensive visualization capabilities

    Parameters:
    - n_clusters: int or None (if None, automatically determines optimal k)
    - random_state: int for reproducibility
    """

    def __init__(self, n_clusters=None, random_state=42):
        """Initialize the clusterer with default parameters"""
        self.n_clusters = n_clusters
        self.random_state = random_state

        # Attributes to be set during fitting
        self.encoder_ = None  # OneHotEncoder for categorical features
        self.sim_matrix_ = None  # Pairwise similarity matrix
        self.feature_clusters_ = None  # KMeans cluster assignments
        self.similarity_clusters_ = None  # Agglomerative cluster assignments
        self.mds_embedding_ = None  # 2D embedding for visualization
        self.X_train_ = None  # Store training data reference
        self.optimal_k_ = None  # Determined optimal clusters

    # --------------------------
    # CORE METHODS
    # --------------------------
    def fit(self, X, y=None):
        """
        Main training method that:
        1. Encodes categorical features
        2. Computes similarity matrix
        3. Determines optimal clusters (if not specified)
        4. Performs dual clustering
        5. Prepares visualization embeddings

        Parameters:
        - X: pandas DataFrame of car features
        - y: unused (for scikit-learn compatibility)
        """
        self.X_train_ = X.copy()

        # 1. Feature Encoding
        self.encoder_ = OneHotEncoder(sparse_output=False)
        X_encoded = self.encoder_.fit_transform(X)

        # 2. Similarity Computation
        self.sim_matrix_ = cosine_similarity(X_encoded)

        # 3. Cluster Determination
        if self.n_clusters is None:
            self.optimal_k_ = self._determine_optimal_k(X_encoded)
            self.n_clusters = self.optimal_k_
        else:
            self.optimal_k_ = self.n_clusters

        # 4. Feature Space Clustering (KMeans)
        kmeans = KMeans(n_clusters=self.n_clusters,
                        random_state=self.random_state)
        self.feature_clusters_ = kmeans.fit_predict(X_encoded)

        # 5. Similarity Space Clustering (Agglomerative)
        distance_matrix = 1 - self.sim_matrix_
        agg = AgglomerativeClustering(n_clusters=self.n_clusters,
                                      metric='precomputed',
                                      linkage='average')
        self.similarity_clusters_ = agg.fit_predict(distance_matrix)

        # 6. Prepare Visualization Embedding (MDS)
        self.mds_embedding_ = MDS(n_components=2,
                                  dissimilarity='precomputed',
                                  random_state=self.random_state).fit_transform(distance_matrix)

        return self

    def _determine_optimal_k(self, X_encoded, max_k=8):
        """
        Internal method to determine optimal number of clusters using elbow method

        Process:
        1. Computes KMeans for k=1 to max_k
        2. Calculates distortions (inertia)
        3. Finds elbow point using second derivatives
        4. Generates and saves elbow plot

        Returns:
        - Optimal number of clusters (int)
        """
        # Compute distortions for each k
        distortions = []
        K_range = range(1, max_k + 1)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            kmeans.fit(X_encoded)
            distortions.append(kmeans.inertia_)

        # Find elbow point (maximum second derivative)
        derivatives = np.diff(distortions, 2)
        optimal_k = np.argmax(derivatives) + 2  # +2 because of second diff

        # Generate elbow plot
        plt.figure(figsize=(10, 5))
        plt.plot(K_range, distortions, 'bx-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Distortion')
        plt.title(f'Elbow Method (Optimal k={optimal_k})')
        plt.axvline(x=optimal_k, color='r', linestyle='--')
        plt.savefig('visualizations/elbow_method.png', bbox_inches='tight')
        plt.close()

        return optimal_k

    def predict(self, X):
        """
        Predict cluster assignments for new data using nearest neighbor approach

        Parameters:
        - X: pandas DataFrame of new car features

        Returns:
        - Array of cluster assignments
        """
        if self.encoder_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Transform new data and compare to training set
        X_encoded = self.encoder_.transform(X)
        sim_to_train = cosine_similarity(X_encoded, self.encoder_.transform(self.X_train_))

        # Assign cluster of most similar training example
        return self.similarity_clusters_[np.argmax(sim_to_train, axis=1)]

    def fit_predict(self, X, y=None):
        """Convenience method for fitting and predicting in one step"""
        self.fit(X)
        return self.similarity_clusters_

    def transform(self, X):
        """
        Transform new data into similarity space

        Returns:
        - Matrix of similarities to training examples
        """
        X_encoded = self.encoder_.transform(X)
        return cosine_similarity(X_encoded, self.encoder_.transform(self.X_train_))

    # --------------------------
    # VISUALIZATION METHODS
    # --------------------------
    def visualize_clusters(self, output_dir='visualizations'):
        """
        Generates and saves all visualizations to specified directory

        Creates:
        1. Cluster comparison plot
        2. Similarity matrix heatmap
        3. Cluster distribution bar plot
        4. Extreme pairs comparison
        5. Similarity distribution histogram
        """
        os.makedirs(output_dir, exist_ok=True)

        # Generate all visualizations
        self._plot_cluster_comparison(output_dir)
        self._plot_similarity_matrix(output_dir)
        self._plot_cluster_distribution(output_dir)
        self._plot_extreme_pairs(output_dir)
        self._plot_similarity_distribution(output_dir)

        print(f"\nAll visualizations saved to {output_dir} directory")

    def _plot_cluster_comparison(self, output_dir):
        """Side-by-side comparison of feature vs similarity clustering"""
        plt.figure(figsize=(15, 6))

        # Feature space clustering
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=self.mds_embedding_[:, 0],
                        y=self.mds_embedding_[:, 1],
                        hue=self.feature_clusters_,
                        palette='viridis', s=100, alpha=0.7)
        plt.title('Feature Space Clustering')

        # Similarity space clustering
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=self.mds_embedding_[:, 0],
                        y=self.mds_embedding_[:, 1],
                        hue=self.similarity_clusters_,
                        palette='viridis', s=100, alpha=0.7)
        plt.title('Similarity Space Clustering')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/cluster_comparison.png', bbox_inches='tight')
        plt.close()

    def _plot_similarity_matrix(self, output_dir):
        """Heatmap of the first 50x50 entries of similarity matrix"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.sim_matrix_[:50, :50], cmap='YlOrRd')
        plt.title('Similarity Matrix (First 50 Samples)')
        plt.savefig(f'{output_dir}/similarity_matrix.png', bbox_inches='tight')
        plt.close()

    def _plot_cluster_distribution(self, output_dir):
        """Bar plot showing count of cars in each cluster"""
        plt.figure(figsize=(10, 5))
        cluster_counts = pd.Series(self.similarity_clusters_).value_counts().sort_index()
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis')
        plt.title('Cluster Distribution')
        plt.xlabel('Cluster')
        plt.ylabel('Count')
        plt.savefig(f'{output_dir}/cluster_distribution.png', bbox_inches='tight')
        plt.close()

    def _plot_extreme_pairs(self, output_dir):
        """Comparison of most and least similar car pairs"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Most similar pair
        max_sim = np.max(self.sim_matrix_ - np.diag(np.diag(self.sim_matrix_)))
        max_indices = np.where(self.sim_matrix_ == max_sim)
        i, j = max_indices[0][0], max_indices[1][0]
        sns.heatmap(self.sim_matrix_[[i, j], :][:, [i, j]],
                    ax=axes[0], cmap='YlOrRd', annot=True, vmin=0, vmax=1)
        axes[0].set_title(f"Most Similar Pair\n(Score: {max_sim:.2f})")

        # Least similar pair
        min_sim = np.min(self.sim_matrix_)
        min_indices = np.where(self.sim_matrix_ == min_sim)
        i, j = min_indices[0][0], min_indices[1][0]
        sns.heatmap(self.sim_matrix_[[i, j], :][:, [i, j]],
                    ax=axes[1], cmap='YlOrRd', annot=True, vmin=0, vmax=1)
        axes[1].set_title(f"Least Similar Pair\n(Score: {min_sim:.2f})")

        plt.tight_layout()
        plt.savefig(f'{output_dir}/extreme_pairs.png', bbox_inches='tight')
        plt.close()

    def _plot_similarity_distribution(self, output_dir):
        """Histogram of pairwise similarity scores"""
        plt.figure(figsize=(10, 5))
        upper_tri = self.sim_matrix_[np.triu_indices_from(self.sim_matrix_, k=1)]
        sns.histplot(upper_tri, bins=30, kde=True)
        plt.title('Distribution of Pairwise Similarities')
        plt.xlabel('Cosine Similarity Score')
        plt.ylabel('Frequency')
        plt.savefig(f'{output_dir}/similarity_distribution.png', bbox_inches='tight')
        plt.close()

    # --------------------------
    # ANALYSIS METHODS
    # --------------------------
    def get_most_similar_pairs(self, n_pairs=3):
        """
        Returns the n most similar car pairs

        Parameters:
        - n_pairs: number of pairs to return

        Returns:
        - List of (index1, index2) tuples
        """
        masked_matrix = self.sim_matrix_.copy()
        np.fill_diagonal(masked_matrix, -1)
        max_indices = np.argsort(masked_matrix.ravel())[-n_pairs * 2:][::-1]
        pairs = [(i // len(self.sim_matrix_), i % len(self.sim_matrix_))
                 for i in max_indices if i // len(self.sim_matrix_) != i % len(self.sim_matrix_)]
        return pairs[:n_pairs]

    def get_least_similar_pairs(self, n_pairs=3):
        """
        Returns the n least similar car pairs

        Parameters:
        - n_pairs: number of pairs to return

        Returns:
        - List of (index1, index2) tuples
        """
        min_sim = np.min(self.sim_matrix_)
        min_indices = np.where(self.sim_matrix_ == min_sim)
        return list(zip(min_indices[0][:n_pairs], min_indices[1][:n_pairs]))


# --------------------------
# EXAMPLE USAGE
# --------------------------
if __name__ == "__main__":
    # Load the car evaluation dataset
    from ucimlrepo import fetch_ucirepo

    car_evaluation = fetch_ucirepo(id=19)
    X = car_evaluation.data.features

    # Initialize and fit the clusterer (auto-determine clusters)
    print("\nInitializing and fitting the clusterer...")
    clusterer = CarSimilarityClusterer(random_state=42)
    clusters = clusterer.fit_predict(X)

    # Add cluster assignments to original data
    X_clustered = X.copy()
    X_clustered['Feature_Cluster'] = clusterer.feature_clusters_
    X_clustered['Similarity_Cluster'] = clusterer.similarity_clusters_

    # Generate all visualizations
    print("\nGenerating visualizations...")
    clusterer.visualize_clusters()

    # Display extreme pairs
    print("\nAnalyzing extreme pairs...")
    print("\nMost similar pairs:")
    for i, j in clusterer.get_most_similar_pairs():
        print(f"\nPair #{i} & #{j}:")
        print(X.iloc[[i, j]])

    print("\nLeast similar pairs:")
    for i, j in clusterer.get_least_similar_pairs():
        print(f"\nPair #{i} & #{j}:")
        print(X.iloc[[i, j]])

    print("\nAnalysis complete!")