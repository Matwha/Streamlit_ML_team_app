"""Helper utilities for the time series forecasting app."""

import streamlit as st
import pandas as pd
from typing import Dict, Optional, Tuple, Any
from config import (TRADITIONAL_MODELS, ML_MODELS, DL_MODELS,
                   MODEL_CATEGORIES)


def setup_environment():
    """Configure logging and environment variables."""
    import os
    import warnings
    import tensorflow as tf

    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    # Configure TensorFlow
    tf.get_logger().setLevel('ERROR')

    # Suppress other warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)


def validate_data(df: pd.DataFrame, min_points: int = 100) -> Tuple[bool, str]:
    """
    Validate input data.

    Args:
        df: Input dataframe
        min_points: Minimum required data points

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if df is None:
        return False, "No data provided"

    if len(df) < min_points:
        return False, f"Dataset too small (minimum {min_points} points required)"

    return True, "Data validation successful"


def setup_model_selection() -> Tuple[Optional[str], Dict]:
    """
    Setup model selection interface in sidebar.
    Returns:
        Tuple[str, Dict]: Selected model name and parameters
    """
    st.sidebar.header("Model Selection")

    # Model category selection
    selected_category = st.sidebar.selectbox(
        "Select Model Category",
        MODEL_CATEGORIES,
        index=0
    )

    # Build available models list based on category
    if selected_category == 'Traditional':
        available_models = TRADITIONAL_MODELS
    elif selected_category == 'Machine Learning':
        available_models = ML_MODELS
    elif selected_category == 'Deep Learning':
        available_models = DL_MODELS
    else:
        available_models = []

    if not available_models:
        st.warning("No models available for selected category.")
        return [], None

    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Model",
        available_models
    )

    return available_models, model_name


def get_model_parameters(model_name: str) -> Dict:
    """
    Get model parameters based on model type.

    Args:
        model_name: Name of the model

    Returns:
        Dict: Model parameters
    """
    params = {
        'build_params': {},
        'train_params': {}
    }
    with st.sidebar.expander(f"{model_name} Parameters", expanded=True):
        if model_name in ['Simple RNN', 'LSTM', 'Stacked LSTM+RNN', 'Sequential']:
            # Model building parameters
            params['build_params']['sequence_length'] = st.slider('Sequence Length', 5, 50, 10)

            if model_name == 'Simple RNN':
                params['build_params']['units'] = st.slider('RNN Units', 8, 128, 32, 8)
                params['build_params']['dropout_rate'] = st.slider('Dropout Rate', 0.0, 0.5, 0.1, 0.1)

            elif model_name == 'LSTM':
                params['build_params']['units'] = st.slider('LSTM Units', 8, 128, 64, 8)
                params['build_params']['dropout_rate'] = st.slider('Dropout Rate', 0.0, 0.5, 0.2, 0.1)

            elif model_name == 'Stacked LSTM+RNN':
                params['build_params']['lstm_units'] = st.slider('LSTM Units', 32, 256, 128, 16)
                params['build_params']['rnn_units'] = st.slider('RNN Units', 16, 128, 64, 8)
                params['build_params']['dropout_rate'] = st.slider('Dropout Rate', 0.0, 0.5, 0.2, 0.1)

            elif model_name == 'Sequential':
                n_layers = st.slider('Number of Layers', 1, 5, 2)
                for i in range(n_layers):
                    layer_params = {}
                    layer_params['type'] = st.selectbox(f'Layer {i + 1} Type',
                                                        ['Dense', 'LSTM', 'GRU'],
                                                        key=f'layer_{i}_type')
                    layer_params['units'] = st.slider(f'{layer_params["type"]} Units',
                                                      8, 128, 32, 8,
                                                      key=f'layer_{i}_units')
                    layer_params['activation'] = st.selectbox(f'Layer {i + 1} Activation',
                                                              ['relu', 'tanh', 'sigmoid'],
                                                              key=f'layer_{i}_activation')
                    layer_params['dropout'] = st.slider(f'Layer {i + 1} Dropout',
                                                        0.0, 0.5, 0.1, 0.1,
                                                        key=f'layer_{i}_dropout')
                    params['build_params'][f'layer_{i}'] = layer_params

            # Training parameters
            params['train_params']['batch_size'] = st.select_slider('Batch Size',
                                                                    options=[16, 32, 64, 128],
                                                                    value=32)
            params['train_params']['epochs'] = st.slider('Training Epochs', 10, 200, 50)
            params['train_params']['learning_rate'] = st.slider('Learning Rate',
                                                                0.0001, 0.01, 0.001,
                                                                format="%.4f")
            params['train_params']['optimizer'] = st.selectbox('Optimizer',
                                                               ['adam', 'rmsprop', 'sgd'])

        # Other model parameters remain the same...
        elif model_name == 'ARIMA':
            params['build_params'].update({
                'p': st.slider('AR order (p)', 0, 5, 1),
                'd': st.slider('Differencing order (d)', 0, 2, 1),
                'q': st.slider('MA order (q)', 0, 5, 1)
            })
        elif model_name == 'SARIMA':
            params = {
                'p': st.slider('AR order (p)', 0, 5, 1),
                'd': st.slider('Differencing order (d)', 0, 2, 1),
                'q': st.slider('MA order (q)', 0, 5, 1),
                'P': st.slider('Seasonal AR order (P)', 0, 2, 1),
                'D': st.slider('Seasonal differencing (D)', 0, 1, 1),
                'Q': st.slider('Seasonal MA order (Q)', 0, 2, 1),
                's': st.select_slider('Seasonal period (s)',
                                    options=[4, 7, 12], value=12)
            }
        elif model_name == 'Random Forest':
            params = {
                'n_estimators': st.slider('Number of Trees', 10, 200, 100),
                'max_depth': st.slider('Max Tree Depth', 2, 30, 10),
                'min_samples_split': st.slider('Min Samples Split', 2, 20, 2),
                'random_state': 42
            }
        elif model_name == 'XGBoost':
            params = {
                'n_estimators': st.slider('Number of Trees', 10, 200, 100),
                'max_depth': st.slider('Max Tree Depth', 2, 30, 6),
                'learning_rate': st.slider('Learning Rate', 0.01, 0.3, 0.1, 0.01),
                'random_state': 42
            }
        elif model_name in ['Simple RNN', 'LSTM', 'Stacked LSTM+RNN', 'Sequential']:
            params = {}

            # Common parameters for all DL models
            params['sequence_length'] = st.slider('Sequence Length', 5, 50, 10)
            params['batch_size'] = st.select_slider('Batch Size',
                                                    options=[16, 32, 64, 128], value=32)
            params['epochs'] = st.slider('Training Epochs', 10, 200, 50)

            # Model specific parameters
            if model_name == 'Simple RNN':
                params['units'] = st.slider('RNN Units', 8, 128, 32, 8)
                params['dropout_rate'] = st.slider('Dropout Rate', 0.0, 0.5, 0.1, 0.1)

            elif model_name == 'LSTM':
                params['units'] = st.slider('LSTM Units', 8, 128, 64, 8)
                params['dropout_rate'] = st.slider('Dropout Rate', 0.0, 0.5, 0.2, 0.1)

            elif model_name == 'Stacked LSTM+RNN':
                params['lstm_units'] = st.slider('LSTM Units', 32, 256, 128, 16)
                params['rnn_units'] = st.slider('RNN Units', 16, 128, 64, 8)
                params['dropout_rate'] = st.slider('Dropout Rate', 0.0, 0.5, 0.2, 0.1)

            elif model_name == 'Sequential':
                n_layers = st.slider('Number of Layers', 1, 5, 2)
                for i in range(n_layers):
                    layer_type = st.selectbox(f'Layer {i + 1} Type',
                                              ['Dense', 'LSTM', 'GRU'], key=f'layer_{i}_type')
                    params[f'layer_{i}_type'] = layer_type
                    params[f'layer_{i}_units'] = st.slider(f'{layer_type} Units',
                                                           8, 128, 32, 8, key=f'layer_{i}_units')
                    params[f'layer_{i}_activation'] = st.selectbox(f'Layer {i + 1} Activation',
                                                                   ['relu', 'tanh', 'sigmoid'],
                                                                   key=f'layer_{i}_activation')
                    params[f'layer_{i}_dropout'] = st.slider(f'Layer {i + 1} Dropout',
                                                             0.0, 0.5, 0.1, 0.1,
                                                             key=f'layer_{i}_dropout')

            # Learning parameters
            params['learning_rate'] = st.slider('Learning Rate', 0.0001, 0.01, 0.001,
                                                format="%.4f")
            params['optimizer'] = st.selectbox('Optimizer', ['adam', 'rmsprop', 'sgd'])

    return params


def display_data_analysis_tabs(analyzer: Any, target_column: str):
    """
    Display data analysis tabs.

    Args:
        analyzer: TimeSeriesAnalyzer instance
        target_column: Name of target column
    """
    tabs = st.tabs([
        "Basic Statistics",
        "Time Series Properties",
        "Data Quality",
        "Feature Engineering"
    ])

    with tabs[0]:
        display_basic_stats(analyzer, target_column)

    with tabs[1]:
        display_time_series_properties(analyzer, target_column)

    with tabs[2]:
        display_data_quality(analyzer, target_column)

    with tabs[3]:
        display_feature_engineering(analyzer, target_column)


def display_basic_stats(analyzer: Any, target_column: str):
    """Display basic statistical analysis."""
    st.subheader("Basic Statistical Analysis")
    stats = analyzer.analyze_basic_stats(target_column)
    st.write(stats)


def display_time_series_properties(analyzer: Any, target_column: str):
    """Display time series properties."""
    st.subheader("Time Series Properties")
    stationarity = analyzer.check_stationarity(target_column)
    seasonality = analyzer.detect_seasonality(target_column)
    st.write("Stationarity:", stationarity)
    st.write("Seasonality:", seasonality)


def display_data_quality(analyzer: Any, target_column: str):
    """Display data quality analysis."""
    st.subheader("Data Quality Analysis")
    missing = analyzer.analyze_missing_values(target_column)
    outliers = analyzer.detect_outliers(target_column)
    st.write("Missing Values:", missing)
    st.write("Outliers:", outliers)


def display_feature_engineering(analyzer: Any, target_column: str):
    """Display feature engineering recommendations."""
    st.subheader("Feature Engineering Recommendations")
    recommendations = analyzer.generate_feature_recommendations(target_column)
    st.write(recommendations)


def create_download_button(df: pd.DataFrame, filename: str = "results.csv"):
    """Create a download button for dataframe."""
    st.download_button(
        label="Download results as CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name=filename,
        mime='text/csv',
    )