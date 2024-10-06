import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.impute import SimpleImputer
from medaxis_core.utils.logging import logger

class ProteomicsDataProcessor:
    def __init__(self, data, scaler=StandardScaler(), imputer=SimpleImputer()):
        self.data = data
        self.scaler = scaler
        self.imputer = imputer

    def preprocess_data(self):
        logger.info("Preprocessing data")
        self.data = self.handle_missing_values()
        self.data = self.scale_data()
        self.data = self.normalize_data()
        self.data = self.encode_data()
        return self.data

    def handle_missing_values(self):
        logger.info("Handling missing values")
        imputed_data = self.imputer.fit_transform(self.data)
        return pd.DataFrame(imputed_data, columns=self.data.columns)

    def scale_data(self):
        logger.info("Scaling data")
        scaled_data = self.scaler.fit_transform(self.data)
        return pd.DataFrame(scaled_data, columns=self.data.columns)

    def normalize_data(self):
        logger.info("Normalizing data")
        normalized_data = self.data / self.data.max()
        return normalized_data

    def encode_data(self):
        logger.info("Encoding data")
        encoded_data = pd.get_dummies(self.data)
        return encoded_data

    def set_scaler(self, scaler):
        self.scaler = scaler

    def set_imputer(self, imputer):
        self.imputer = imputer

    def filter_data(self, threshold=0.5):
        logger.info("Filtering data")
        filtered_data = self.data[self.data['expression_level'] > threshold]
        return filtered_data

    def transform_data(self, transformation='log2'):
        logger.info("Transforming data")
        if transformation == 'log2':
            transformed_data = np.log2(self.data)
        elif transformation == 'log10':
            transformed_data = np.log10(self.data)
        else:
            logger.warning("Invalid transformation method")
            return self.data
        return pd.DataFrame(transformed_data, columns=self.data.columns)

    def reduce_dimensions(self, method='pca', n_components=2):
        logger.info("Reducing dimensions")
        if method == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components)
            reduced_data = pca.fit_transform(self.data)
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=n_components)
            reduced_data = tsne.fit_transform(self.data)
        else:
            logger.warning("Invalid dimensionality reduction method")
            return self.data
        return pd.DataFrame(reduced_data, columns=[f"Component {i+1}" for i in range(n_components)])
