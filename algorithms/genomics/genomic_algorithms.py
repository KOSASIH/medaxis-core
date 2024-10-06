import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from medaxis_core.utils.logging import logger
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection

class GenomicsAlgorithms:
    def __init__(self, data, target, test_size=0.2, random_state=42):
        self.data = data
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self):
        logger.info("Splitting data into training and testing sets")
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, test_size=self.test_size, random_state=self.random_state)
        return X_train, X_test, y_train, y_test

    def scale_data(self, X_train, X_test):
        logger.info("Scaling data using StandardScaler")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def reduce_dimensions(self, X_train_scaled, X_test_scaled, method="pca", n_components=2):
        logger.info(f"Reducing dimensions using {method}")
        if method == "pca":
            pca = PCA(n_components=n_components)
            X_train_reduced = pca.fit_transform(X_train_scaled)
            X_test_reduced = pca.transform(X_test_scaled)
        elif method == "tsne":
            tsne = TSNE(n_components=n_components)
            X_train_reduced = tsne.fit_transform(X_train_scaled)
            X_test_reduced = tsne.transform(X_test_scaled)
        elif method == "nmf":
            nmf = NMF(n_components=n_components)
            X_train_reduced = nmf.fit_transform(X_train_scaled)
            X_test_reduced = nmf.transform(X_test_scaled)
        elif method == "truncated_svd":
            truncated_svd = TruncatedSVD(n_components=n_components)
            X_train_reduced = truncated_svd.fit_transform(X_train_scaled)
            X_test_reduced = truncated_svd.transform(X_test_scaled)
        elif method == "gaussian_random_projection":
            gaussian_random_projection = GaussianRandomProjection(n_components=n_components)
            X_train_reduced = gaussian_random_projection.fit_transform(X_train_scaled)
            X_test_reduced = gaussian_random_projection.transform(X_test_scaled)
        elif method == "sparse_random_projection":
            sparse_random_projection = SparseRandomProjection(n_components=n_components)
            X_train_reduced = sparse_random_projection.fit_transform(X_train_scaled)
            X_test_reduced = sparse_random_projection.transform(X_test_scaled)
        else:
            logger.warning("Invalid dimensionality reduction method")
            return X_train_scaled, X_test_scaled
        return X_train_reduced, X_test_reduced

    def select_features(self, X_train_scaled, y_train, method="kbest", k=10):
        logger.info(f"Selecting features using {method}")
        if method == "kbest":
            selector = SelectKBest(f_classif, k=k)
            X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        elif method == "mutual_info":
            selector = SelectKBest(mutual_info_classif, k=k)
            X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        else:
            logger.warning("Invalid feature selection method")
            return X_train_scaled
        return X_train_selected

    def cluster_data(self, X_train_reduced, method="kmeans", n_clusters=2):
        logger.info(f"Clustering data using {method}")
        if method == "kmeans":
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(X_train_reduced)
            labels = kmeans.labels_
        else:
            logger.warning("Invalid clustering method")
            return None
        return labels

    def evaluate_clustering(self, X_train_reduced, labels):
        logger.info("Evaluating clustering performance")
        silhouette = silhouette_score(X_train_reduced, labels)
        calinski_harabasz = calinski_harabasz_score(X_train_reduced, labels)
        davies_bouldin = davies_bouldin_score(X_train_reduced, labels)
        return silhouette, calinski_harabasz, davies_bouldin

    def train_model(self, X_train_scaled, y_train, model_type="random_forest"):
        logger.info(f"Training {model_type} model")
        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        elif model_type == "svm":
            model = SVC(kernel="rbf", C=1, random_state=self.random_state)
        elif model_type == "logistic_regression":
            model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        else:
            logger.warning("Invalid model type")
            return None
        model.fit(X_train_scaled, y_train)
        return model

    def tune_model(self, X_train_scaled, y_train, model_type="random_forest"):
        logger.info(f"Tuning {model_type} model")
        if model_type == "random_forest":
            param_grid = {
                "n_estimators": [10, 50, 100, 200],
                "max_depth": [None, 5, 10, 15],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 5, 10]
            }
            model = RandomForestClassifier(random_state=self.random_state)
        elif model_type == "svm":
            param_grid = {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf", "poly"],
                "gamma": ["scale", "auto"]
            }
            model = SVC(random_state=self.random_state)
        elif model_type == "logistic_regression":
            param_grid = {
                "C": [0.1, 1, 10],
                "penalty": ["l1", "l2"],
                "max_iter": [100, 500, 1000]
            }
            model = LogisticRegression(random_state=self.random_state)
        else:
            logger.warning("Invalid model type")
            return None
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
        grid_search.fit(X_train_scaled, y_train)
        return grid_search.best_estimator_

    def evaluate_model(self, model, X_test_scaled, y_test):
        logger.info("Evaluating model performance")
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred)
        return accuracy, report, matrix

    def run_pipeline(self, model_type="random_forest", method="pca", n_components=2, n_clusters=2, k=10):
        X_train, X_test, y_train, y_test = self.split_data()
        X_train_scaled, X_test_scaled = self.scale_data(X_train, X_test)
        X_train_reduced, X_test_reduced = self.reduce_dimensions(X_train_scaled, X_test_scaled, method, n_components)
        X_train_selected = self.select_features(X_train_reduced, y_train, method="kbest", k=k)
        labels = self.cluster_data(X_train_selected, method="kmeans", n_clusters=n_clusters)
        silhouette, calinski_harabasz, davies_bouldin = self.evaluate_clustering(X_train_selected, labels)
        model = self.train_model(X_train_selected, y_train, model_type)
        accuracy, report, matrix = self.evaluate_model(model, X_test_scaled, y_test)
        logger.info(f"Model accuracy: {accuracy:.3f}")
        logger.info(f"Classification report:\n{report}")
        logger.info(f"Confusion matrix:\n{matrix}")
        logger.info(f"Silhouette score: {silhouette:.3f}")
        logger.info(f"Calinski-Harabasz score: {calinski_harabasz:.3f}")
        logger.info(f"Davies-Bouldin score: {davies_bouldin:.3f}")
        return accuracy, report, matrix, silhouette, calinski_harabasz, davies_bouldin

    def run_tuned_pipeline(self, model_type="random_forest", method="pca", n_components=2, n_clusters=2, k=10):
        X_train, X_test, y_train, y_test = self.split_data()
        X_train_scaled, X_test_scaled = self.scale_data(X_train, X_test)
        X_train_reduced, X_test_reduced = self.reduce_dimensions(X_train_scaled, X_test_scaled, method, n_components)
        X_train_selected = self.select_features(X_train_reduced, y_train, method ="kbest", k=k)
        labels = self.cluster_data(X_train_selected, method="kmeans", n_clusters=n_clusters)
        silhouette, calinski_harabasz, davies_bouldin = self.evaluate_clustering(X_train_selected, labels)
        model = self.tune_model(X_train_selected, y_train, model_type)
        accuracy, report, matrix = self.evaluate_model(model, X_test_scaled, y_test)
        logger.info(f"Model accuracy: {accuracy:.3f}")
        logger.info(f"Classification report:\n{report}")
        logger.info(f"Confusion matrix:\n{matrix}")
        logger.info(f"Silhouette score: {silhouette:.3f}")
        logger.info(f"Calinski-Harabasz score: {calinski_harabasz:.3f}")
        logger.info(f"Davies-Bouldin score: {davies_bouldin:.3f}")
        return accuracy, report, matrix, silhouette, calinski_harabasz, davies_bouldin
