"""Configuration settings and constants for the time series forecasting app."""

# App settings
APP_TITLE = "Time Series Forecasting Application"
APP_DESCRIPTION = """
This app performs univariate time series forecasting using various models including traditional, 
machine learning and deep learning approaches. Features include:
- Multiple model selection and comparison
- Comprehensive visualization options
- Confidence interval generation
- Cross-validation capabilities
- Model persistence
"""

# Default data sources
DEFAULT_GITHUB_REPO = "https://github.com/PJalgotrader/Deep_forecasting-USU/tree/main/data"
GITHUB_REPO_URL = "https://github.com/PJalgotrader/Deep_forecasting-USU/tree/main/data"
GITHUB_RAW_BASE_URL = "https://raw.githubusercontent.com/PJalgotrader/Deep_forecasting-USU/main/data"
DEFAULT_EXAMPLE_FILE = "airline_passengers.csv"

# Model categories and options
MODEL_CATEGORIES = ['Traditional', 'Machine Learning', 'Deep Learning']
TRADITIONAL_MODELS = ['ARIMA', 'SARIMA']
ML_MODELS = ['Random Forest', 'XGBoost']
DL_MODELS = ['Simple RNN', 'LSTM', 'Stacked LSTM+RNN', 'Sequential']

# Visualization settings
VIZ_BACKEND_OPTIONS = ['Matplotlib', 'Plotly']
PLOT_TYPES = [
    'Time Series Plot',
    'Seasonal Decomposition',
    'Correlation Plot',
    'Distribution Plot',
    'Prediction vs Actual',
    'Residuals Analysis',
    'Feature Importance',
    'Model Performance Metrics',
    'Training History',
    'Confidence Intervals'
]

# Confidence interval methods
CI_METHODS = {
    'Traditional': ['Analytic', 'Bootstrap'],
    'Machine Learning': ['Bootstrap', 'Quantile'],
    'Deep Learning': ['Monte Carlo Dropout', 'Bootstrap']
}

# Cross validation settings
CV_SPLITS = 5
CV_METHODS = ['TimeSeriesSplit', 'Blocked TimeSeriesSplit', 'Rolling Forecast']

# Model parameters defaults
# Traditional Models
ARIMA_DEFAULTS = {
    'p_max': 5,
    'd_max': 2,
    'q_max': 5,
    'seasonal': True
}

SARIMA_DEFAULTS = {
    'p_max': 5,
    'd_max': 2,
    'q_max': 5,
    'P_max': 2,
    'D_max': 1,
    'Q_max': 2,
    'seasonal_periods': [4, 7, 12]
}

# Machine Learning Models
RF_DEFAULTS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'bootstrap': True
}

XGBOOST_DEFAULTS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

# Deep Learning Models
DL_DEFAULTS = {
    'sequence_length': 60,
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'dropout_rate': 0.2,
    'validation_split': 0.2
}

# Data processing settings
MIN_TRAINING_POINTS = 100
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VAL_SIZE = 0.2
RANDOM_SEED = 42

# File paths
MODEL_SAVE_PATH = "saved_models/"
RESULTS_SAVE_PATH = "results/"
LOG_PATH = "logs/"

# Data source settings
EXAMPLE_DATA_URL = "https://raw.githubusercontent.com/PJalgotrader/Deep_forecasting-USU/main/data/US_macro_monthly.csv"

# Error messages
ERROR_MESSAGES = {
    'data_load': "Error loading data: {}",
    'preprocessing': "Error in preprocessing: {}",
    'model_training': "Error training model: {}",
    'validation': "Validation error: {}",
    'prediction': "Error generating predictions: {}",
    'visualization': "Error creating visualization: {}"
}

# Plotting settings
PLOT_STYLE = {
    'figure.figsize': (12, 6),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 2,
    'grid.alpha': 0.3
}

COLORS = {
    'actual': '#2C3E50',
    'predicted': '#E74C3C',
    'confidence': '#3498DB',
    'training': '#2ECC71',
    'validation': '#F1C40F',
    'test': '#9B59B6'
}