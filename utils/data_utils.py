import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

# Define a function to load data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")

# Define a function to preprocess data
def preprocess_data(data):
    try:
        # Drop missing values
        data.dropna(inplace=True)
        
        # Scale data
        scaler = StandardScaler()
        data[['feature1', 'feature2', 'feature3']] = scaler.fit_transform(data[['feature1', 'feature2', 'feature3']])
        
        return data
    except Exception as e:
        print(f"Error preprocessing data: {e}")

# Define a function to split data into training and testing sets
def split_data(data, test_size=0.2, random_state=42):
    try:
        X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error splitting data: {e}")

# Define a function to reduce dimensionality using PCA
def reduce_dimensionality_pca(data, n_components=2):
    try:
        pca = PCA(n_components=n_components)
        data_reduced = pca.fit_transform(data)
        return data_reduced
    except Exception as e:
        print(f"Error reducing dimensionality using PCA: {e}")

# Define a function to reduce dimensionality using t-SNE
def reduce_dimensionality_tsne(data, n_components=2):
    try:
        tsne = TSNE(n_components=n_components)
        data_reduced = tsne.fit_transform(data)
        return data_reduced
    except Exception as e:
        print(f"Error reducing dimensionality using t-SNE: {e}")

# Define a function to cluster data using K-Means
def cluster_data(data, n_clusters=2):
    try:
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(data)
        labels = kmeans.labels_
        return labels
    except Exception as e:
        print(f"Error clustering data: {e}")

# Define a function to evaluate clustering performance
def evaluate_clustering(data, labels):
    try:
        silhouette = silhouette_score(data, labels)
        calinski_harabasz = calinski_harabasz_score(data, labels)
        davies_bouldin = davies_bouldin_score(data, labels)
        return silhouette, calinski_harabasz, davies_bouldin
    except Exception as e:
        print(f"Error evaluating clustering performance: {e}")

# Define a function to visualize data
def visualize_data(data, labels):
    try:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette='viridis')
        plt.title('Data Visualization')
        plt.show()
    except Exception as e:
        print(f"Error visualizing data: {e}")
