import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from medaxis_core.utils.logging import logger

class ProteomicsDataLoader:
    def __init__(self, file_path, metadata_file_path=None, test_size=0.2, random_state=42):
        self.file_path = file_path
        self.metadata_file_path = metadata_file_path
        self.test_size = test_size
        self.random_state = random_state

    def load_data(self):
        logger.info(f"Loading data from {self.file_path}")
        data = pd.read_csv(self.file_path)
        return data

    def load_metadata(self):
        if self.metadata_file_path:
            logger.info(f"Loading metadata from {self.metadata_file_path}")
            metadata = pd.read_csv(self.metadata_file_path)
            return metadata
        else:
            logger.warning("No metadata file provided")
            return None

    def split_data(self, data, metadata=None):
        logger.info("Splitting data into training and testing sets")
        X = data.drop(['protein_id'], axis=1)
        y = data['protein_id']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        if metadata:
            metadata_train, metadata_test = train_test_split(metadata, test_size=self.test_size, random_state=self.random_state)
            return X_train, X_test, y_train, y_test, metadata_train, metadata_test
        else:
            return X_train, X_test, y_train, y_test

    def shuffle_data(self, data):
        logger.info("Shuffling data")
        return shuffle(data, random_state=self.random_state)

    def get_data_stats(self, data):
        logger.info("Calculating data statistics")
        stats = data.describe()
        return stats

    def get_data_correlation_matrix(self, data):
        logger.info("Calculating data correlation matrix")
        corr_matrix = data.corr()
        return corr_matrix

    def filter_data(self, data, threshold=0.5):
        logger.info("Filtering data")
        filtered_data = data[data['expression_level'] > threshold]
        return filtered_data

    def normalize_data(self, data):
        logger.info("Normalizing data")
        normalized_data = data / data.max()
        return normalized_data

    def encode_data(self, data):
        logger.info("Encoding data")
        encoded_data = pd.get_dummies(data)
        return encoded_data
