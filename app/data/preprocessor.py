"""Data preprocessing and dataset creation functions."""

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict
from sklearn.preprocessing import StandardScaler
import streamlit as st
from sklearn.model_selection import TimeSeriesSplit


class TimeSeriesPreprocessor:
    """Enhanced class for preprocessing time series data."""

    def __init__(self):
        """Initialize preprocessor with default settings."""
        self.scaler = None
        # Add default preprocessing options
        self.preprocessing_options = {
            'min_data_points': 100,
            'max_missing_pct': 20,
            'train_size': 0.8,
            'validation_split': True,
            'val_size': 0.2,
            'handle_missing': 'Forward Fill',
            'handle_outliers': False,
            'outlier_method': 'IQR',
            'scaling_method': 'StandardScaler',
            'transform_method': 'None',
            'add_date_features': True,
            'create_lags': True,
            'max_lags': 7,
            'min_lags': 1,
            'add_rolling_features': True,
            'rolling_windows': [7, 14]
        }

    def add_preprocessing_controls(self):
        """Add preprocessing control options to Streamlit sidebar."""
        st.sidebar.header("Data Preprocessing")

        options = {}

        with st.sidebar.expander("Data Validation", expanded=True):
            options['min_data_points'] = st.number_input(
                "Minimum Required Data Points",
                min_value=30,
                max_value=1000,
                value=100,
                help="Minimum number of data points required for modeling"
            )

            options['max_missing_pct'] = st.slider(
                "Maximum Missing Values (%)",
                min_value=0,
                max_value=50,
                value=20,
                help="Maximum percentage of missing values allowed"
            )

        with st.sidebar.expander("Data Splitting", expanded=True):
            options['train_size'] = st.slider(
                "Training Data Size (%)",
                min_value=50,
                max_value=90,
                value=80,
                help="Percentage of data to use for training"
            ) / 100.0

            options['validation_split'] = st.checkbox(
                "Use Validation Split",
                value=True,
                help="Split training data into training and validation sets"
            )

            if options['validation_split']:
                options['val_size'] = st.slider(
                    "Validation Data Size (%)",
                    min_value=10,
                    max_value=30,
                    value=20,
                    help="Percentage of training data to use for validation"
                ) / 100.0

        with st.sidebar.expander("Data Cleaning", expanded=True):
            options['handle_missing'] = st.selectbox(
                "Handle Missing Values",
                options=['Forward Fill', 'Backward Fill', 'Linear Interpolation', 'Mean', 'None'],
                help="Method to handle missing values"
            )

            options['handle_outliers'] = st.checkbox(
                "Remove Outliers",
                value=False,
                help="Remove statistical outliers from the data"
            )

            if options['handle_outliers']:
                options['outlier_method'] = st.selectbox(
                    "Outlier Detection Method",
                    options=['IQR', 'Z-Score', 'Isolation Forest'],
                    help="Method to detect outliers"
                )

        with st.sidebar.expander("Scaling & Transformation", expanded=True):
            options['scaling_method'] = st.selectbox(
                "Scaling Method",
                options=['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'None'],
                help="Method to scale the data"
            )

            options['transform_method'] = st.selectbox(
                "Data Transformation",
                options=['None', 'Log', 'Square Root', 'Box-Cox'],
                help="Transform data to handle non-linearity or non-normality"
            )

        with st.sidebar.expander("Feature Engineering", expanded=True):
            options['add_date_features'] = st.checkbox(
                "Add Date Features",
                value=True,
                help="Add year, month, day features for datetime index"
            )

            options['create_lags'] = st.checkbox(
                "Create Lag Features",
                value=True,
                help="Create lagged versions of the target variable"
            )

            if options['create_lags']:
                options['max_lags'] = st.slider(
                    "Maximum Lag Period",
                    min_value=1,
                    max_value=30,
                    value=7,
                    help="Maximum number of lag periods to create"
                )

                options['min_lags'] = st.slider(
                    "Minimum Required Periods",
                    min_value=1,
                    max_value=options['max_lags'],
                    value=1,
                    help="Minimum number of lag periods required"
                )

            options['add_rolling_features'] = st.checkbox(
                "Add Rolling Statistics",
                value=True,
                help="Add rolling mean and standard deviation"
            )

            if options['add_rolling_features']:
                options['rolling_windows'] = st.multiselect(
                    "Rolling Window Sizes",
                    options=[3, 7, 14, 30],
                    default=[7, 14],
                    help="Window sizes for rolling statistics"
                )

        self.preprocessing_options = options
        return options

    def prepare_data(self, df: pd.DataFrame, target_col: str, model_type: str) -> Dict:
        """Prepare data according to selected preprocessing options."""
        try:
            # Initial validation checks
            if len(df) < self.preprocessing_options['min_data_points']:
                st.error(f"Insufficient data points. Minimum required: {self.preprocessing_options['min_data_points']}")
                return None

            missing_pct = df[target_col].isnull().mean() * 100
            if missing_pct > self.preprocessing_options['max_missing_pct']:
                st.error(
                    f"Too many missing values ({missing_pct:.1f}%). Maximum allowed: {self.preprocessing_options['max_missing_pct']}%")
                return None

            # Make a copy to avoid modifying original data
            processed_df = df.copy()

            # Handle missing values first if selected
            if self.preprocessing_options['handle_missing'] != 'None':
                processed_df = self._handle_missing_values(processed_df, target_col)
                if processed_df is None:
                    return None

            # Handle outliers if selected
            if self.preprocessing_options.get('handle_outliers', False):
                processed_df = self._handle_outliers(
                    processed_df,
                    target_col,
                    method=self.preprocessing_options['outlier_method']
                )
                if processed_df is None:
                    return None

            # Apply transformation if selected
            if self.preprocessing_options['transform_method'] != 'None':
                processed_df = self._transform_data(
                    processed_df,
                    target_col,
                    method=self.preprocessing_options['transform_method']
                )
                if processed_df is None:
                    return None


            # Scale data if selected
            if self.preprocessing_options['scaling_method'] != 'None':
                processed_df = self._scale_data(processed_df, target_col,
                                                self.preprocessing_options['scaling_method'])

            # Add features if selected
            if self.preprocessing_options['add_date_features']:
                processed_df = self._add_date_features(processed_df)

            if self.preprocessing_options.get('create_lags', False):
                processed_df = self._create_lag_features(
                    processed_df, target_col,
                    self.preprocessing_options['max_lags']
                )

            if self.preprocessing_options.get('add_rolling_features', False):
                processed_df = self._add_rolling_features(processed_df, target_col)

            # Split size calculations
            train_size = int(len(processed_df) * self.preprocessing_options['train_size'])

            if self.preprocessing_options['validation_split']:
                val_size = int(train_size * self.preprocessing_options['val_size'])
                train_size = train_size - val_size
            else:
                val_size = 0

            # Create splits based on model type
            if model_type in ['ARIMA', 'SARIMA']:
                splits = {
                    'train': processed_df[target_col][:train_size],
                    'test': processed_df[target_col][train_size + val_size:],
                    'test_target': processed_df[target_col][train_size + val_size:],
                    'test_index': processed_df.index[train_size + val_size:]
                }
                if val_size > 0:
                    splits['val'] = processed_df[target_col][train_size:train_size + val_size]
            else:
                # For ML/DL models
                feature_cols = [col for col in processed_df.columns if col != target_col]
                splits = {
                    'train': (processed_df[feature_cols][:train_size],
                              processed_df[target_col][:train_size]),
                    'test': (processed_df[feature_cols][train_size + val_size:],
                             processed_df[target_col][train_size + val_size:]),
                    'test_target': processed_df[target_col][train_size + val_size:],
                    'test_index': processed_df.index[train_size + val_size:]
                }
                if val_size > 0:
                    splits['val'] = (processed_df[feature_cols][train_size:train_size + val_size],
                                     processed_df[target_col][train_size:train_size + val_size])

            return splits

        except Exception as e:
            st.error(f"Error in data preparation: {str(e)}")
            return None

    def _handle_missing_values(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Handle missing values using specified method."""
        method = self.preprocessing_options['handle_missing']

        if method == 'Forward Fill':
            df[target_col] = df[target_col].ffill()
        elif method == 'Backward Fill':
            df[target_col] = df[target_col].bfill()
        elif method == 'Linear Interpolation':
            df[target_col] = df[target_col].interpolate()
        elif method == 'Mean':
            df[target_col] = df[target_col].fillna(df[target_col].mean())

        return df

    def _scale_data(self, df: pd.DataFrame, target_col: str, method: str) -> pd.DataFrame:
        """Scale data using specified method."""
        if method == 'StandardScaler':
            self.scaler = StandardScaler()
        elif method == 'MinMaxScaler':
            self.scaler = MinMaxScaler()
        elif method == 'RobustScaler':
            self.scaler = RobustScaler()

        df[target_col] = self.scaler.fit_transform(df[target_col].values.reshape(-1, 1))
        return df

    # Add this method to the TimeSeriesPreprocessor class in preprocessor.py

    def prepare_sequences(self, data: pd.Series, sequence_length: int, n_features: int = 1,
                          train_size: float = 0.8) -> Dict:
        """Prepare sequences for RNN/LSTM models.

        Args:
            data: Pandas Series containing the target variable
            sequence_length: Length of input sequences
            n_features: Number of features (1 for univariate)
            train_size: Proportion of data for training

        Returns:
            Dict containing train/val/test splits as numpy arrays
        """
        try:
            # Convert data to numpy array if needed
            values = data.values if isinstance(data, pd.Series) else data
            values = values.reshape(-1, 1) if n_features == 1 else values

            # Create sequences
            sequences = []
            targets = []

            for i in range(len(values) - sequence_length):
                sequences.append(values[i:(i + sequence_length)])
                targets.append(values[i + sequence_length])

            # Convert to numpy arrays
            X = np.array(sequences)
            y = np.array(targets)

            # Reshape if needed
            if n_features == 1:
                X = X.reshape((X.shape[0], X.shape[1], 1))
                y = y.reshape(-1)

            # Calculate split sizes
            train_idx = int(len(X) * train_size)
            val_idx = train_idx
            if self.preprocessing_options.get('validation_split', True):
                val_size = self.preprocessing_options.get('val_size', 0.2)
                val_idx = train_idx - int(train_idx * val_size)

            # Create splits
            splits = {
                'train': (X[:val_idx], y[:val_idx])
            }

            if self.preprocessing_options.get('validation_split', True):
                splits['val'] = (X[val_idx:train_idx], y[val_idx:train_idx])

            splits.update({
                'test': (X[train_idx:], y[train_idx:]),
                'test_target': pd.Series(y[train_idx:], index=data.index[train_idx + sequence_length:]),
                'test_index': data.index[train_idx + sequence_length:]
            })

            return splits

        except Exception as e:
            st.error(f"Error preparing sequences: {str(e)}")
            return None

    def _prepare_standard_split(self, df: pd.DataFrame, target_col: str) -> Dict:
        """Prepare train/val/test split for traditional models."""
        train_size = int(len(df) * self.preprocessing_options['train_size'])

        if self.preprocessing_options['validation_split']:
            val_size = int(train_size * self.preprocessing_options['val_size'])
            train_size = train_size - val_size

            return {
                'train': df[target_col][:train_size],
                'val': df[target_col][train_size:train_size + val_size],
                'test': df[target_col][train_size + val_size:],
                'scaler': self.scaler
            }
        else:
            return {
                'train': df[target_col][:train_size],
                'test': df[target_col][train_size:],
                'scaler': self.scaler
            }

    def _add_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add date-based features if index is datetime."""
        if isinstance(df.index, pd.DatetimeIndex):
            df['year'] = df.index.year
            df['month'] = df.index.month
            df['day'] = df.index.day
            df['dayofweek'] = df.index.dayofweek
            df['quarter'] = df.index.quarter
        return df

    def _create_lag_features(self, df: pd.DataFrame, target_col: str, max_lags: int) -> pd.DataFrame:
        """Create lagged versions of target variable."""
        for lag in range(1, max_lags + 1):
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        return df.dropna()

    def _add_rolling_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add rolling mean and standard deviation features."""
        for window in [7, 14, 30]:
            df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df[target_col].rolling(window=window).std()
        return df.dropna()

    def calculate_prediction_intervals(self, model, X, confidence=0.95, n_iterations=100):
        """Calculate prediction intervals using bootstrap."""
        try:
            predictions = []
            is_sequence = len(X.shape) == 3  # Check if input is sequential data

            for _ in range(n_iterations):
                # Bootstrap the data
                if is_sequence:
                    indices = np.random.choice(len(X), size=len(X), replace=True)
                    X_bootstrap = X[indices]
                else:
                    if isinstance(X, pd.DataFrame):
                        X_bootstrap = X.sample(n=len(X), replace=True)
                    else:
                        indices = np.random.choice(len(X), size=len(X), replace=True)
                        X_bootstrap = X[indices]

                # Generate predictions
                pred = model.predict(X_bootstrap)
                if hasattr(self, 'scaler') and self.scaler is not None:
                    pred = self.scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
                predictions.append(pred)

            # Calculate intervals
            predictions = np.array(predictions)
            lower = np.percentile(predictions, ((1 - confidence) / 2) * 100, axis=0)
            upper = np.percentile(predictions, (1 - (1 - confidence) / 2) * 100, axis=0)
            mean_pred = np.mean(predictions, axis=0)

            return mean_pred, lower, upper

        except Exception as e:
            st.error(f"Error calculating prediction intervals: {str(e)}")
            return None, None, None

    def create_cross_validation_splits(self, X, y, n_splits=5):
        """Create time series cross validation splits."""
        try:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            splits = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                splits.append(((X_train, y_train), (X_val, y_val)))

            return splits

        except Exception as e:
            st.error(f"Error creating CV splits: {str(e)}")
            return None


