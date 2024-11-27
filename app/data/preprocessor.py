"""Data preprocessing and dataset creation functions."""
"""Enhanced data preprocessing with prediction intervals and feature engineering."""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import streamlit as st
from sklearn.model_selection import TimeSeriesSplit


class TimeSeriesPreprocessor:
    """Enhanced class for preprocessing time series data."""

    def __init__(self):
        self.scaler = StandardScaler()

    def create_features(self, df, target_col, lags=None, add_date_features=True):
        """Create features for ML models."""
        try:
            features = pd.DataFrame(index=df.index)

            # Add lag features
            if lags is None:
                lags = [1, 7, 14, 30]  # Default lags

            for lag in lags:
                features[f'lag_{lag}'] = df[target_col].shift(lag)

            # Add date features if requested
            if add_date_features and isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
                features['year'] = df.index.year
                features['month'] = df.index.month
                features['day'] = df.index.day
                features['day_of_week'] = df.index.dayofweek
                features['quarter'] = df.index.quarter

            # Add rolling statistics
            for window in [7, 14, 30]:
                features[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
                features[f'rolling_std_{window}'] = df[target_col].rolling(window=window).std()

            # Add target column
            features[target_col] = df[target_col]

            return features.dropna()

        except Exception as e:
            st.error(f"Error creating features: {str(e)}")
            return None

    def prepare_data(self, series):
        """
        Prepare the raw data with normalization and scaling.
        """
        try:
            # Debug logging
            st.write("Debug: Initial series type:", type(series))
            st.write("Debug: Initial series dtype:", series.dtype)
            st.write("Debug: First few values:", series.head())

            # Ensure we're working with numeric data
            if not pd.api.types.is_numeric_dtype(series):
                st.error("Target variable must be numeric")
                return None, None

            # Convert to numpy array
            data = series.values

            # Ensure data is 2D
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
                st.write("Debug: Reshaped data shape:", data.shape)

            # Handle inf and nan values
            data = np.nan_to_num(data, nan=np.nanmean(data), posinf=None, neginf=None)
            st.write("Debug: Data shape before scaling:", data.shape)
            st.write("Debug: Data min/max:", np.min(data), np.max(data))

            # Standardize the data
            try:
                scaled_data = self.scaler.fit_transform(data)
                st.write("Debug: Scaling successful")
                st.write("Debug: Scaled data shape:", scaled_data.shape)
                return scaled_data, self.scaler
            except Exception as e:
                st.error(f"Error during scaling: {str(e)}")
                return None, None

        except Exception as e:
            st.error(f"Error in data preparation: {str(e)}")
            st.write("Debug: Full error:", str(e))
            return None, None

    def prepare_ml_data(self, features, target_col, test_size=0.2):
        """Prepare data for ML models."""
        try:
            # Split features and target
            y = features[target_col]
            X = features.drop(columns=[target_col])

            # Split into train and test
            split_idx = int(len(features) * (1 - test_size))

            X_train = X[:split_idx]
            X_test = X[split_idx:]
            y_train = y[:split_idx]
            y_test = y[split_idx:]

            return (X_train, y_train), (X_test, y_test)

        except Exception as e:
            st.error(f"Error preparing ML data: {str(e)}")
            return None

    def calculate_prediction_intervals(self, model, X, confidence=0.95, n_iterations=100):
        """Calculate prediction intervals using bootstrap."""
        try:
            predictions = []

            for _ in range(n_iterations):
                # Bootstrap the data
                indices = np.random.choice(len(X), size=len(X), replace=True)
                X_bootstrap = X[indices]

                # Generate predictions
                if hasattr(model, 'predict'):
                    pred = model.predict(X_bootstrap)
                else:
                    pred = model(X_bootstrap)

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

    def prepare_multivariate_data(self, df, target_cols, sequence_length, batch_size):
        """Prepare data for multivariate time series."""
        try:
            # Scale data
            scaled_data = self.scaler.fit_transform(df[target_cols])

            # Create sequences
            X, y = [], []
            for i in range(len(scaled_data) - sequence_length):
                X.append(scaled_data[i:(i + sequence_length)])
                y.append(scaled_data[i + sequence_length])

            X = np.array(X)
            y = np.array(y)

            # Create TF datasets
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
            dataset = dataset.shuffle(1000).batch(batch_size)

            return dataset, self.scaler

        except Exception as e:
            st.error(f"Error preparing multivariate data: {str(e)}")
            return None, None

    def create_sequences(self, scaled_data, sequence_length, batch_size, val_split=0.2, test_split=0.2):
        """Create TensorFlow datasets with proper sizing based on sequences_count"""
        try:
            st.write("Debug: Starting create_sequences")
            st.write("Debug: scaled_data shape:", scaled_data.shape)
            st.write("Debug: sequence_length:", sequence_length)
            st.write("Debug: batch_size:", batch_size)

            n = len(scaled_data)
            sequences_count = n - sequence_length

            # Ensure enough data points
            if sequences_count < 3:
                st.error(
                    f"Not enough sequences for the given sequence length ({sequence_length}). Minimum required is 3 sequences, but got {sequences_count}.")
                return None, None, None

            # Calculate the effective sizes based on sequences_count
            train_size = int(sequences_count * (1 - val_split - test_split))
            val_size = int(sequences_count * val_split)
            test_size = sequences_count - train_size - val_size

            # Adjust proportions if any split is too small
            min_split_size = 1  # At least one sequence per split
            if train_size < min_split_size or val_size < min_split_size or test_size < min_split_size:
                st.warning("Adjusting dataset splits to ensure all splits have at least one sample.")
                train_size = max(min_split_size, int(sequences_count * 0.7))
                val_size = max(min_split_size, int(sequences_count * 0.15))
                test_size = sequences_count - train_size - val_size

            # Prepare sequences and targets
            sequences = []
            targets = []
            for i in range(sequences_count):
                sequences.append(scaled_data[i:(i + sequence_length)])
                targets.append(scaled_data[i + sequence_length])

            sequences = np.array(sequences)
            targets = np.array(targets)

            # Split the sequences and targets
            train_sequences = sequences[:train_size]
            train_targets = targets[:train_size]

            val_sequences = sequences[train_size:train_size + val_size]
            val_targets = targets[train_size:train_size + val_size]

            test_sequences = sequences[train_size + val_size:]
            test_targets = targets[train_size + val_size:]

            # Handle cases where any dataset split is empty
            if len(train_sequences) == 0 or len(val_sequences) == 0 or len(test_sequences) == 0:
                st.error(
                    "One of the datasets (train, val, or test) is empty. Adjust the sequence length or split ratios.")
                return None, None, None

            # Adjust batch size based on train_size
            adjusted_batch_size = max(1, min(batch_size, train_size // 10))

            # Create TensorFlow datasets
            train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_targets)).shuffle(1024).batch(
                adjusted_batch_size)
            val_dataset = tf.data.Dataset.from_tensor_slices((val_sequences, val_targets)).batch(adjusted_batch_size)
            test_dataset = tf.data.Dataset.from_tensor_slices((test_sequences, test_targets)).batch(adjusted_batch_size)

            return train_dataset, val_dataset, test_dataset

        except Exception as e:
            st.error(f"Error creating datasets: {str(e)}")
            return None, None, None

    def create_tf_datasets(self, sequences, targets, batch_size, val_split=0.2, test_split=0.2):
        """
        Create TensorFlow datasets for training, validation, and testing.

        Args:
            sequences (np.array): Sequence data
            targets (np.array): Target values
            batch_size (int): Batch size
            val_split (float): Validation split ratio
            test_split (float): Test split ratio

        Returns:
            tuple: (train_dataset, val_dataset, test_dataset) or (None, None, None) if error
        """
        try:
            # Calculate split indices
            n = len(sequences)
            train_size = int(n * (1 - val_split - test_split))
            val_size = int(n * val_split)

            # Split data
            train_sequences = sequences[:train_size]
            train_targets = targets[:train_size]

            val_sequences = sequences[train_size:train_size + val_size]
            val_targets = targets[train_size:train_size + val_size]

            test_sequences = sequences[train_size + val_size:]
            test_targets = targets[train_size + val_size:]

            # Create TF datasets
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (train_sequences, train_targets)).shuffle(1024).batch(batch_size)
            val_dataset = tf.data.Dataset.from_tensor_slices(
                (val_sequences, val_targets)).batch(batch_size)
            test_dataset = tf.data.Dataset.from_tensor_slices(
                (test_sequences, test_targets)).batch(batch_size)

            return train_dataset, val_dataset, test_dataset

        except Exception as e:
            st.error(f"Error creating TF datasets: {str(e)}")
            return None, None, None

    def prepare_dl_datasets(self, series, sequence_length, batch_size, val_split=0.2, test_split=0.2):
        """Main function to prepare data for deep learning with validation"""
        try:
            # Convert index to datetime if not already
            if not isinstance(series.index, pd.DatetimeIndex):
                try:
                    series.index = pd.to_datetime(series.index)
                except Exception as e:
                    st.warning(f"Could not convert index to datetime: {str(e)}")

            # Check for NaN values
            if series.isna().any():
                st.error("Data contains missing values. Please clean the data first.")
                return None, None, None, None

            # Check if series is too short
            if len(series) < sequence_length + 3:  # At least 3 sequences
                st.error(f"Time series too short. Need at least {sequence_length + 3} points.")
                return None, None, None, None

            # Create progress containers
            prep_status = st.empty()
            prep_progress = st.progress(0)

            # Data preparation steps
            prep_status.text("Preparing data...")
            prep_progress.progress(0.2)

            scaled_data, scaler = self.prepare_data(series)
            if scaled_data is None:
                prep_progress.empty()
                prep_status.empty()
                return None, None, None, None
            prep_progress.progress(0.6)

            prep_status.text("Creating datasets...")
            train_dataset, val_dataset, test_dataset = self.create_sequences(
                scaled_data=scaled_data,
                sequence_length=sequence_length,
                batch_size=batch_size,
                val_split=val_split,
                test_split=test_split
            )

            if train_dataset is None:
                st.error("Failed to create datasets. Please adjust the parameters.")
                prep_progress.empty()
                prep_status.empty()
                return None, None, None, None

            prep_progress.progress(1.0)
            prep_status.text("Data preparation complete!")

            # Clean up progress indicators
            prep_status.empty()
            prep_progress.empty()

            return train_dataset, val_dataset, test_dataset, scaler

        except Exception as e:
            st.error(f"Error in data preparation: {str(e)}")
            return None, None, None, None