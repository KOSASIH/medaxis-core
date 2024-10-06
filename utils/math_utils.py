import numpy as np
import math
from scipy.stats import norm
from scipy.optimize import minimize

# Define a function to calculate the mean of a list of numbers
def calculate_mean(numbers):
    try:
        mean = np.mean(numbers)
        return mean
    except Exception as e:
        print(f"Error calculating mean: {e}")

# Define a function to calculate the standard deviation of a list of numbers
def calculate_std_dev(numbers):
    try:
        std_dev = np.std(numbers)
        return std_dev
    except Exception as e:
        print(f"Error calculating standard deviation: {e}")

# Define a function to calculate the variance of a list of numbers
def calculate_variance(numbers):
    try:
        variance = np.var(numbers)
        return variance
    except Exception as e:
        print(f"Error calculating variance: {e}")

# Define a function to calculate the median of a list of numbers
def calculate_median(numbers):
    try:
        median = np.median(numbers)
        return median
    except Exception as e:
        print(f"Error calculating median: {e}")

# Define a function to calculate the mode of a list of numbers
def calculate_mode(numbers):
    try:
        mode = np.bincount(numbers).argmax()
        return mode
    except Exception as e:
        print(f"Error calculating mode: {e}")

# Define a function to calculate the range of a list of numbers
def calculate_range(numbers):
    try:
        range_ = np.ptp(numbers)
        return range_
    except Exception as e:
        print(f"Error calculating range: {e}")

# Define a function to calculate the interquartile range (IQR) of a list of numbers
def calculate_iqr(numbers):
    try:
        q75, q25 = np.percentile(numbers, [75, 25])
        iqr = q75 - q25
        return iqr
    except Exception as e:
        print(f"Error calculating IQR: {e}")

# Define a function to calculate the skewness of a list of numbers
def calculate_skewness(numbers):
    try:
        skewness = np.mean((numbers - np.mean(numbers)) ** 3) / np.std(numbers) ** 3
        return skewness
    except Exception as e:
        print(f"Error calculating skewness: {e}")

# Define a function to calculate the kurtosis of a list of numbers
def calculate_kurtosis(numbers):
    try:
        kurtosis = np.mean((numbers - np.mean(numbers)) ** 4) / np.std(numbers) ** 4 - 3
        return kurtosis
    except Exception as e:
        print(f"Error calculating kurtosis: {e}")

# Define a function to calculate the correlation coefficient between two lists of numbers
def calculate_correlation(numbers1, numbers2):
    try:
        correlation = np.corrcoef(numbers1, numbers2)[0, 1]
        return correlation
    except Exception as e:
        print(f"Error calculating correlation: {e}")

# Define a function to calculate the regression line between two lists of numbers
def calculate_regression_line(numbers1, numbers2):
    try:
        slope, intercept = np.polyfit(numbers1, numbers2, 1)
        return slope, intercept
    except Exception as e:
        print(f"Error calculating regression line: {e}")

# Define a function to calculate the probability density function (PDF) of a normal distribution
def calculate_pdf(x, mean, std_dev):
    try:
        pdf = norm.pdf(x, mean, std_dev)
        return pdf
    except Exception as e:
        print(f"Error calculating PDF: {e}")

# Define a function to calculate the cumulative distribution function (CDF) of a normal distribution
def calculate_cdf(x, mean, std_dev):
    try:
        cdf = norm.cdf(x, mean, std_dev)
        return cdf
    except Exception as e:
        print(f"Error calculating CDF: {e}")

# Define a function to calculate the inverse CDF of a normal distribution
def calculate_inverse_cdf(p, mean, std_dev):
    try:
        inverse_cdf = norm.ppf(p, mean, std_dev)
        return inverse_cdf
    except Exception as e:
        print(f"Error calculating inverse CDF: {e}")

# Define a function to minimize a function using the minimize function from scipy
def minimize_function(func, x0, method='BFGS'):
    try:
        res = minimize(func, x0, method=method)
        return res.x
    except Exception as e:
        print(f"Error minimizing function: {e}")
