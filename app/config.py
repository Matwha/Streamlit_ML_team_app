# config.py
APP_TITLE = "Time Series Forecasting Application"

AVAILABLE_METRICS = [
    'MAE', 'RMSE', 'MAPE', 'R-squared',
    'Training Time', 'Prediction Speed',
    'AIC', 'BIC', 'Cross-validation Score'
]

DEFAULT_METRICS = ['MAE', 'RMSE', 'MAPE']

ALL_MODELS = [
    'ARIMA', 'SARIMA',
    'Random Forest', 'XGBoost',
    'Simple RNN', 'LSTM', 'Stacked LSTM+RNN'
]

MODEL_CONFIGS = {
    'ARIMA': {
        'preprocessing': ['differencing', 'log_transform'],
        'ci_methods': ['analytical', 'bootstrap', 'quantile'],
        'cv_method': 'time_series_split'
    },
    'SARIMA': {
        'preprocessing': ['seasonal_decompose', 'differencing'],
        'ci_methods': ['analytical', 'bootstrap', 'quantile'],
        'cv_method': 'time_series_split'
    },
    'Random Forest': {
        'preprocessing': ['scaling', 'lag_features'],
        'ci_methods': ['bootstrap', 'quantile', 'conformal'],
        'cv_method': 'blocked_time_series'
    },
    'XGBoost': {
        'preprocessing': ['scaling', 'lag_features'],
        'ci_methods': ['bootstrap', 'quantile', 'conformal'],
        'cv_method': 'blocked_time_series'
    },
    'Simple RNN': {
        'preprocessing': ['scaling', 'sequence_creation'],
        'ci_methods': ['monte_carlo', 'bootstrap'],
        'architectures': ['vanilla', 'bidirectional', 'stacked']
    },
    'LSTM': {
        'preprocessing': ['scaling', 'sequence_creation'],
        'ci_methods': ['monte_carlo', 'bootstrap'],
        'architectures': ['vanilla', 'bidirectional', 'stacked']
    },
    'Stacked LSTM+RNN': {
        'preprocessing': ['scaling', 'sequence_creation'],
        'ci_methods': ['monte_carlo', 'bootstrap'],
        'architectures': ['parallel', 'sequential']
    }
}

# Training configurations
TRAINING_CONFIGS = {
    'deep_learning': {
        'batch_sizes': [16, 32, 64, 128],
        'learning_rates': [1e-4, 1e-3, 1e-2],
        'optimizers': ['adam', 'rmsprop', 'sgd'],
        'activations': ['relu', 'tanh', 'sigmoid']
    },
    'ml': {
        'cv_splits': range(2, 11),
        'scoring': ['neg_mean_absolute_error', 'neg_mean_squared_error']
    }
}

# Visualization settings
PLOT_CONFIGS = {
    'prediction': {
        'actual_color': 'blue',
        'pred_color': 'red',
        'ci_color': 'rgba(0,100,80,0.2)'
    }
}