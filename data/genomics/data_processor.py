import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.impute import SimpleImputer
from medaxis_core.utils.logging import logger

class GenomicsDataProcessor:
    def __init__(self, data, scaler=StandardScaler(), imputer=SimpleImputer()):
        self.data = data
        self.scaler = scaler
        self.imputer = imputer

    def preprocess_data(self):
        logger.info("Preprocessing data")
        self.data = self.handle_missing_values()
        self.data = self.scale_data()
        return self.data

    def handle_missing_values(self):
        logger.info("Handling missing values")
        imputed_data = self.imputer.fit_transform(self.data)
        return pd.DataFrame(imputed_data, columns=self.data.columns)

    def scale_data(self):
        logger.info("Scaling data")
        scaled_data = self.scaler.fit_transform(self.data)
        return pd.DataFrame(scaled_data, columns=self.data.columns)

    def set_scaler(self, scaler):
        self.scaler = scaler

    def set_imputer(self, imputer):
        self.imputer = imputer
