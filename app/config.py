"""Configuration settings and constants for the time series forecasting app."""


# App settings
APP_TITLE = "Time Series Forecasting Application"
APP_DESCRIPTION = """
This app performs univariate time series forecasting using various models including deep learning.
Models available:
- **Traditional Models:** ARIMA, SARIMA
- **Machine Learning:** Random Forest, XGBoost
- **Deep Learning:** RNN, LSTM, Stacked LSTM+RNN
"""

# Model categories and options
MODEL_CATEGORIES = ['Traditional', 'Machine Learning', 'Deep Learning']
TRADITIONAL_MODELS = ['ARIMA', 'SARIMA']
ML_MODELS = ['Random Forest', 'XGBoost']
DL_MODELS = ['Simple RNN', 'LSTM', 'Stacked LSTM+RNN']

# Default parameters
DEFAULT_SEQUENCE_LENGTH = 60
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_RNN_UNITS = 64
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_TEST_SPLIT = 0.2

# Data settings
MIN_DATA_POINTS = 100
DATA_SOURCE_OPTIONS = ["GitHub Repository", "Upload File", "Example Data"]
DEFAULT_GITHUB_REPO = "https://github.com/PJalgotrader/Deep_forecasting-USU/tree/main/data"
EXAMPLE_DATA_URL = "https://raw.githubusercontent.com/PJalgotrader/Deep_forecasting-USU/main/data/yfinance.csv"
