import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, hstack, vstack, csc_matrix
import warnings
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings('ignore')


# =============================================
# 1. Data Preprocessing Class
# =============================================
class TransactionPreprocessor:
    def __init__(self, chunk_size=50000):
        self.chunk_size = chunk_size
        self.vectorizer = TfidfVectorizer(max_features=500)
        self.scaler = StandardScaler()

    def process_product_text(self, product_str):
        """Convert product lists to clean strings"""
        try:
            if isinstance(product_str, str):
                # Safely evaluate the string representation of list
                products = product_str.strip("[]").replace("'", "").split(", ")
            else:
                products = product_str
            return ' '.join(products).lower()
        except:
            return 'unknown'

    def preprocess_chunk(self, chunk):
        """Process a single chunk of data"""
        # Text features
        chunk['clean_products'] = chunk['Product'].apply(self.process_product_text)

        # Categorical features
        categoricals = chunk[[
            'Store_Type', 'City', 'Payment_Method',
            'Customer_Category', 'Season', 'Promotion'
        ]].fillna('unknown')

        # Numerical features
        numericals = chunk[['Total_Items', 'Total_Cost']].fillna(0)

        return {
            'text': chunk['clean_products'],
            'categorical': categoricals,
            'numerical': numericals
        }


# =============================================
# 2. Clustering Engine with Similarity
# =============================================
class TransactionClusterer:
    def __init__(self, n_clusters=8, random_state=42, chunk_size=50000):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.chunk_size = chunk_size
        self.preprocessor = TransactionPreprocessor(chunk_size)
        self.model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=chunk_size
        )
        self.feature_matrix = None

    def plot_similarity_matrix(self, sim_matrix, output_dir='visualizations'):
        """Visualize a portion of the similarity matrix with high resolution"""
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)

        # Create larger figure with higher DPI
        plt.figure(figsize=(16, 14), dpi=800)

        # Show first 50x50 or full matrix if smaller
        display_size = min(50, sim_matrix.shape[0])

        # Create heatmap with better parameters
        ax = sns.heatmap(sim_matrix[:display_size, :display_size],
                         cmap='YlOrRd',
                         xticklabels=False,
                         yticklabels=False,
                         square=True,
                         cbar_kws={'shrink': 0.8})

        # Add title with better formatting
        plt.title('Transaction Similarity Matrix (First 50 Samples)\n',
                  fontsize=14, fontweight='bold', pad=20)

        # Improve color bar visibility
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)

        # Adjust layout and save
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'similarity_matrix.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=False)
        plt.close()

        print(f"High-resolution similarity matrix saved to: {save_path}")


    def _transform_features(self, processed):
        """Transform features to sparse matrix format"""
        text_features = self.preprocessor.vectorizer.transform(processed['text'])
        num_features = self.preprocessor.scaler.transform(processed['numerical'])
        cat_features = pd.get_dummies(processed['categorical'])

        # Convert all to sparse matrices for efficient storage
        return hstack([
            text_features,
            csr_matrix(num_features),
            csr_matrix(cat_features.values)
        ])

    def fit(self, file_path):
        """Train the model on large data in chunks"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")

        try:
            # Initialize CSV reader
            reader = pd.read_csv(file_path, chunksize=self.chunk_size)

            # First pass: Fit transformers
            print("Fitting feature transformers...")
            first_chunk = next(reader)
            processed = self.preprocessor.preprocess_chunk(first_chunk)

            self.preprocessor.vectorizer.fit(processed['text'])
            self.preprocessor.scaler.fit(processed['numerical'])

            # Second pass: Train clusters (rewind file)
            reader = pd.read_csv(file_path, chunksize=self.chunk_size)
            print("Training clustering model...")

            for chunk in tqdm(reader, desc="Processing chunks"):
                processed = self.preprocessor.preprocess_chunk(chunk)
                features = self._transform_features(processed)
                self.model.partial_fit(features)

            return self

        except Exception as e:
            print(f"Error during fitting: {str(e)}")
            raise

    def predict(self, file_path):
        """Generate predictions in chunks"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")

        results = []
        feature_matrices = []
        try:
            reader = pd.read_csv(file_path, chunksize=self.chunk_size)

            for chunk in tqdm(reader, desc="Predicting clusters"):
                processed = self.preprocessor.preprocess_chunk(chunk)
                features = self._transform_features(processed)

                preds = self.model.predict(features)
                chunk['Cluster'] = preds
                results.append(chunk)
                feature_matrices.append(features)

            self.feature_matrix = vstack(feature_matrices) if len(feature_matrices) > 0 else None
            return pd.concat(results)

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise

    def calculate_cosine_similarity(self, sample_size=1000, random_state=42):
        """Calculate cosine similarity between transactions"""
        if self.feature_matrix is None:
            raise ValueError("No feature matrix available. Run predict() first.")

        # Sample a subset for efficient calculation
        np.random.seed(random_state)
        n = min(sample_size, self.feature_matrix.shape[0])
        sampled_indices = np.random.choice(self.feature_matrix.shape[0], n, replace=False)
        sampled_matrix = self.feature_matrix[sampled_indices]

        # Calculate cosine similarity
        print(f"Calculating cosine similarity for {n} samples...")
        similarity_matrix = cosine_similarity(sampled_matrix)

        # Plot the matrix
        self.plot_similarity_matrix(similarity_matrix)

        return similarity_matrix, sampled_indices


# =============================================
# 3. Visualization Helper Class
# =============================================
class VisualizationHelper:
    @staticmethod
    def plot_cluster_distribution(clustered_data, output_dir='visualizations'):
        """Plot distribution of transactions across clusters"""
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(12, 6))
        cluster_counts = clustered_data['Cluster'].value_counts().sort_index()
        ax = sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis')

        plt.title('Transaction Cluster Distribution\n', fontsize=14, fontweight='bold')
        plt.xlabel('Cluster ID', fontsize=12)
        plt.ylabel('Number of Transactions', fontsize=12)
        ax.bar_label(ax.containers[0], fmt='%d')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_distribution.png'), dpi=300)
        plt.close()
        print(f"Cluster distribution plot saved to: {os.path.join(output_dir, 'cluster_distribution.png')}")

    @staticmethod
    def plot_numerical_features(clustered_data, output_dir='visualizations'):
        """Plot distribution of numerical features per cluster"""
        os.makedirs(output_dir, exist_ok=True)

        numerical_features = ['Total_Items', 'Total_Cost']

        for feature in numerical_features:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=clustered_data, x='Cluster', y=feature, palette='viridis')
            plt.title(f'{feature} Distribution by Cluster\n', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{feature.lower()}_distribution.png'), dpi=300)
            plt.close()
            print(f"Numerical feature plot saved: {feature.lower()}_distribution.png")

    @staticmethod
    def plot_categorical_features(clustered_data, output_dir='visualizations'):
        """Plot distribution of categorical features per cluster"""
        os.makedirs(output_dir, exist_ok=True)
        categoricals = ['Store_Type', 'Payment_Method', 'Customer_Category']

        for feature in categoricals:
            plt.figure(figsize=(14, 8))
            cluster_props = clustered_data.groupby(['Cluster', feature]).size().unstack()
            cluster_props = cluster_props.div(cluster_props.sum(axis=1), axis=0)

            cluster_props.plot(kind='bar', stacked=True, colormap='viridis')
            plt.title(f'{feature} Distribution by Cluster\n', fontsize=14, fontweight='bold')
            plt.xlabel('Cluster', fontsize=12)
            plt.ylabel('Proportion', fontsize=12)
            plt.legend(title=feature, bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{feature.lower()}_distribution.png'), dpi=300)
            plt.close()
            print(f"Categorical feature plot saved: {feature.lower()}_distribution.png")

    @staticmethod
    def plot_cluster_projection(feature_matrix, clusters, output_dir='visualizations', sample_size=1000):
        """2D projection of clusters using PCA"""
        from sklearn.decomposition import PCA

        os.makedirs(output_dir, exist_ok=True)

        # Sample for efficient visualization
        if feature_matrix.shape[0] > sample_size:
            indices = np.random.choice(feature_matrix.shape[0], sample_size, replace=False)
            sampled_features = feature_matrix[indices]
            sampled_clusters = clusters.iloc[indices]
        else:
            sampled_features = feature_matrix
            sampled_clusters = clusters

        # Convert sparse matrix to dense for PCA
        if isinstance(sampled_features, (csr_matrix, csc_matrix)):
            sampled_features = sampled_features.toarray()

        # Reduce dimensionality
        pca = PCA(n_components=2)
        projections = pca.fit_transform(sampled_features)

        plt.figure(figsize=(14, 10))
        sns.scatterplot(x=projections[:, 0], y=projections[:, 1],
                        hue=sampled_clusters, palette='viridis',
                        alpha=0.7, s=50, edgecolor='none')

        plt.title('2D Cluster Projection (PCA)\n', fontsize=14, fontweight='bold')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_projection.png'), dpi=300)
        plt.close()
        print(f"Cluster projection plot saved: cluster_projection.png")

    @staticmethod
    def plot_product_wordclouds(clustered_data, output_dir='visualizations', max_words=50):
        """Generate word clouds of products for each cluster"""
        from wordcloud import WordCloud

        os.makedirs(output_dir, exist_ok=True)

        for cluster in sorted(clustered_data['Cluster'].unique()):
            cluster_text = ' '.join(clustered_data[clustered_data['Cluster'] == cluster]['clean_products'])

            wordcloud = WordCloud(width=800, height=400,
                                  background_color='white',
                                  max_words=max_words).generate(cluster_text)

            plt.figure(figsize=(14, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(f'Most Common Products - Cluster {cluster}\n', fontsize=14, fontweight='bold')
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'wordcloud_cluster_{cluster}.png'), dpi=300)
            plt.close()
            print(f"Word cloud saved for cluster {cluster}")


# =============================================
# 4. Main Execution with Enhanced Visualization
# =============================================
if __name__ == "__main__":
    try:
        # Configuration
        INPUT_FILE = os.path.abspath("Retail_Transactions_Dataset.csv")
        OUTPUT_FILE = os.path.abspath("clustered_transactions.csv")
        VIS_DIR = "visualizations"
        SIMILARITY_FILE = os.path.abspath("transaction_similarity.npy")
        N_CLUSTERS = 10
        CHUNK_SIZE = 100000
        SAMPLE_SIZE = 1000

        print(f"Looking for input file at: {INPUT_FILE}")
        if not os.path.exists(INPUT_FILE):
            raise FileNotFoundError(f"Input file not found at {INPUT_FILE}")

        # Run pipeline
        print("Starting clustering pipeline...")
        start_time = time.time()

        # 1. Initialize components
        clusterer = TransactionClusterer(n_clusters=N_CLUSTERS, chunk_size=CHUNK_SIZE)
        visualizer = VisualizationHelper()

        # 2. Train model
        clusterer.fit(INPUT_FILE)

        # 3. Generate predictions
        clustered_data = clusterer.predict(INPUT_FILE)
        clustered_data.to_csv(OUTPUT_FILE, index=False)
        print(f"Results saved to: {OUTPUT_FILE}")

        # 4. Generate visualizations
        print("\nGenerating visualizations...")
        visualizer.plot_cluster_distribution(clustered_data, VIS_DIR)
        visualizer.plot_numerical_features(clustered_data, VIS_DIR)
        visualizer.plot_categorical_features(clustered_data, VIS_DIR)

        if clusterer.feature_matrix is not None:
            # Dimensionality reduction plot
            visualizer.plot_cluster_projection(clusterer.feature_matrix,
                                               clustered_data['Cluster'],
                                               VIS_DIR)

            # Product analysis
            visualizer.plot_product_wordclouds(clustered_data, VIS_DIR)

            # Similarity analysis
            similarity_matrix, _ = clusterer.calculate_cosine_similarity(SAMPLE_SIZE)
            np.save(SIMILARITY_FILE, similarity_matrix)
            print(f"Similarity matrix saved to: {SIMILARITY_FILE}")

        # 5. Final summary
        print(f"\nClustering completed in {time.time() - start_time:.2f} seconds")
        print("\nCluster distribution summary:")
        print(clustered_data['Cluster'].value_counts().sort_index())

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Verify the input file exists at the specified path")
        print("2. Check file permissions")
        print("3. Ensure you have enough disk space")
        print("4. Try reducing chunk_size if memory is limited")