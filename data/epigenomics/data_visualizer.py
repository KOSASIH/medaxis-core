import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from medaxis_core.utils.logging import logger

class EpigenomicsDataVisualizer:
    def __init__(self, data):
        self.data = data

    def plot_histogram(self, column, bins=50):
        logger.info(f"Plotting histogram for column {column}")
        plt.hist(self.data[column], bins=bins)
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()

    def plot_scatterplot(self, x_column, y_column):
        logger.info(f"Plotting scatterplot for columns {x_column} and {y_column}")
        sns.scatterplot(x=self.data[x_column], y=self.data[y_column])
        plt.title(f"Scatterplot of {x_column} vs {y_column}")
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()

    def plot_barplot(self, column):
        logger.info(f"Plotting barplot for column {column}")
        sns.barplot(x=self.data[column].value_counts().index, y=self.data[column].value_counts())
        plt.title(f"Barplot of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()

    def plot_heatmap(self, columns):
        logger.info(f"Plotting heatmap for columns {columns}")
        corr_matrix = self.data[columns].corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", square=True)
        plt.title(f"Heatmap of {columns}")
        plt.show()

    def plot_boxplot(self, column):
        logger.info(f"Plotting boxplot for column {column}")
        sns.boxplot(self.data[column])
        plt.title(f"Boxplot of {column}")
        plt.xlabel(column)
        plt.ylabel("Value")
        plt.show()

    def plot_violinplot(self, column):
        logger.info(f"Plotting violinplot for column {column}")
        sns.violinplot(self.data[column])
        plt.title(f"Violinplot of {column}")
        plt.xlabel(column)
        plt.ylabel("Value")
        plt.show()

    def plot_pairplot(self, columns):
        logger.info(f"Plotting pairplot for columns {columns}")
        sns.pairplot(self.data[columns])
        plt.title(f"Pairplot of {columns}")
        plt.show()

    def plot_clustermap(self, columns):
        logger.info(f"Plotting clustermap for columns {columns}")
        sns.clustermap(self.data[columns].corr(), annot=True, cmap="coolwarm", square=True)
        plt.title(f"Clustermap of {columns}")
        plt.show()

    def plot_methylation_levels(self, chromosome, position):
        logger.info(f"Plotting methylation levels for chromosome {chromosome} and position {position}")
        methylation_levels = self.data[(self.data['chromosome'] == chromosome) & (self.data['position'] == position)]['methylation_level']
        sns.boxplot(methylation_levels)
        plt.title(f"Methylation levels of chromosome {chromosome} and position {position}")
        plt.xlabel("Methylation level")
        plt.ylabel("Value")
        plt.show()

    def plot_chromosome_methylation(self, chromosome):
        logger.info(f"Plotting chromosome methylation for chromosome {chromosome}")
        chromosome_methylation = self.data[self.data['chromosome'] == chromosome]['methylation_level']
        sns.boxplot(chromosome_methylation)
        plt.title(f"Chromosome methylation of {chromosome}")
        plt.xlabel("Methylation level")
        plt.ylabel("Value")
        plt.show()

    def plot_position_methylation(self, position):
        logger.info(f"Plotting position methylation for position {position}")
        position_methylation = self.data[self.data['position'] == position]['methylation_level']
        sns.boxplot(position_methylation)
        plt.title(f"Position methylation of {position}")
        plt.xlabel("Methylation level")
        plt.ylabel("Value")
        plt.show()
