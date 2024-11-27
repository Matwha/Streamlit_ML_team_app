"""Helper utilities for the time series forecasting app."""

import streamlit as st
import pandas as pd


def validate_data(df, min_points=100):
    """
    Validate input data.

    Args:
        df: Input dataframe or series
        min_points: Minimum required data points

    Returns:
        bool: Whether data is valid
    """
    if df is None:
        return False

    if isinstance(df, pd.DataFrame):
        num_rows = df.shape[0]
    else:
        num_rows = len(df)

    if num_rows < min_points:
        st.error(f"Dataset too small for reliable training (minimum {min_points} points required)")
        return False

    return True


def get_time_index(df):
    """Get appropriate time index for the dataset."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index
    else:
        try:
            first_col = df.iloc[:, 0]
            if pd.to_datetime(first_col, errors='coerce').notnull().all():
                return pd.to_datetime(first_col)
        except:
            pass
    return np.arange(len(df))


def setup_logging():
    """Configure logging and warning settings."""
    import warnings
    import os
    import tensorflow as tf

    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')


def create_download_button(df, filename="results.csv"):
    """Create a download button for dataframe."""
    st.download_button(
        label="Download results as CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name=filename,
        mime='text/csv',
    )


def setup_environment():
    """Configure logging and environment variables."""
    import os
    import warnings
    import tensorflow as tf

    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

    # Configure TensorFlow
    tf.get_logger().setLevel('ERROR')

    # Suppress other warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)