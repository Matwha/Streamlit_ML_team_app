# models/rnn_lstm.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from abc import abstractmethod
import matplotlib.pyplot as plt
from sklearn.utils import resample
from scipy import stats


class TrainingMonitor:
    """Complete training monitoring system for deep learning models."""

    def __init__(self):
        """Initialize monitoring components."""
        # Create placeholders for all visualization elements
        self.loss_plot = st.empty()
        self.metrics_plot = st.empty()
        self.progress_bar = st.progress(0)
        self.status = st.empty()
        self.batch_metrics = st.empty()
        self.validation_metrics = st.empty()

        # Store history for plotting
        self.history = {
            'loss': [],
            'val_loss': [],
            'metrics': {}
        }

        # Create container for epoch metrics
        self.epoch_metrics_container = st.container()

    def on_epoch_begin(self, epoch, num_epochs):
        """Update status at start of epoch."""
        self.status.text(f"Epoch {epoch + 1}/{num_epochs}")
        self.progress_bar.progress((epoch + 1) / num_epochs)

    def on_batch_end(self, batch, logs):
        """Update batch-level metrics."""
        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        self.batch_metrics.text(f"Batch {batch}: {metrics_str}")

    def on_epoch_end(self, epoch, logs):
        """Update plots and metrics at end of epoch."""
        # Update history
        for metric, value in logs.items():
            if metric not in self.history['metrics']:
                self.history['metrics'][metric] = []
            if metric == 'loss':
                self.history['loss'].append(value)
            elif metric == 'val_loss':
                self.history['val_loss'].append(value)
            else:
                self.history['metrics'][metric].append(value)

        # Update loss plot
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(y=self.history['loss'], name='Training Loss'))
        if self.history['val_loss']:
            fig_loss.add_trace(go.Scatter(y=self.history['val_loss'], name='Validation Loss'))
        fig_loss.update_layout(title="Loss Over Time", xaxis_title="Epoch", yaxis_title="Loss")
        self.loss_plot.plotly_chart(fig_loss, use_container_width=True)

        # Update metrics plot
        if self.history['metrics']:
            fig_metrics = go.Figure()
            for metric, values in self.history['metrics'].items():
                fig_metrics.add_trace(go.Scatter(y=values, name=metric))
            fig_metrics.update_layout(title="Metrics Over Time", xaxis_title="Epoch", yaxis_title="Value")
            self.metrics_plot.plotly_chart(fig_metrics, use_container_width=True)

        # Update validation metrics
        if any(k.startswith('val_') for k in logs.keys()):
            val_metrics = {k: v for k, v in logs.items() if k.startswith('val_')}
            val_metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            self.validation_metrics.text(f"Validation Metrics: {val_metrics_str}")

    def reset(self):
        """Reset all monitoring components."""
        self.history = {
            'loss': [],
            'val_loss': [],
            'metrics': {}
        }
        self.progress_bar.progress(0)
        self.status.empty()
        self.batch_metrics.empty()
        self.validation_metrics.empty()
        self.loss_plot.empty()
        self.metrics_plot.empty()

class DeepTimeSeriesModel:
    def __init__(self, sequence_length, n_features=1, name='DeepTimeSeriesModel'):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.name = name
        self.model = None
        self.history = None
        self.scaler = None
        self.training_summary = {}

    def prepare_sequences(self, data):
        sequences = []
        targets = []

        for i in range(len(data) - self.sequence_length):
            sequences.append(data[i:(i + self.sequence_length)])
            targets.append(data[i + self.sequence_length])

        return np.array(sequences), np.array(targets)

    def create_callbacks(self, progress_bar, status_text, metrics_container):
        class CustomCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / self.params['epochs']
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch + 1}/{self.params['epochs']}")
                metrics = " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
                metrics_container.text(metrics)

        return [
            CustomCallback(),
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ]

    def train(self, X, y, validation_data=None, **kwargs):
        try:
            history = self.model.fit(
                X, y,
                validation_data=validation_data,
                **kwargs
            )
            self.history = history
            self._update_training_summary()
            return history
        except Exception as e:
            st.error(f"{self.name} training error: {str(e)}")
            return None

    def predict(self, X, return_conf_int=True, alpha=0.05, n_samples=100):
        try:
            pred = self.model.predict(X)

            if return_conf_int:
                # Monte Carlo Dropout for confidence intervals
                predictions = []
                for _ in range(n_samples):
                    predictions.append(self.model(X, training=True))
                predictions = np.array(predictions)

                mean_pred = np.mean(predictions, axis=0)
                std_pred = np.std(predictions, axis=0)

                z_score = stats.norm.ppf(1 - alpha / 2)
                lower = mean_pred - z_score * std_pred
                upper = mean_pred + z_score * std_pred

                return pred, (lower, upper)
            return pred
        except Exception as e:
            st.error(f"{self.name} prediction error: {str(e)}")
            return None

    def _update_training_summary(self):
        self.training_summary = {
            'loss': self.history.history['loss'],
            'val_loss': self.history.history.get('val_loss', []),
            'metrics': {k: v for k, v in self.history.history.items()
                        if k not in ['loss', 'val_loss']}
        }


class SimpleRNNModel(DeepTimeSeriesModel):
    def build(self, units=64, dropout_rate=0.1):
        inputs = keras.Input(shape=(self.sequence_length, self.n_features))
        x = layers.SimpleRNN(units, dropout=dropout_rate)(inputs)
        outputs = layers.Dense(1)(x)

        self.model = keras.Model(inputs, outputs)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return self.model


class LSTMModel(DeepTimeSeriesModel):
    def build(self, units=64, dropout_rate=0.1):
        inputs = keras.Input(shape=(self.sequence_length, self.n_features))
        x = layers.LSTM(units, dropout=dropout_rate)(inputs)
        outputs = layers.Dense(1)(x)

        self.model = keras.Model(inputs, outputs)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return self.model


class StackedModel(DeepTimeSeriesModel):
    def build(self, lstm_units=128, rnn_units=64, dropout_rate=0.1):
        inputs = keras.Input(shape=(self.sequence_length, self.n_features))
        x = layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate)(inputs)
        x = layers.SimpleRNN(rnn_units, dropout=dropout_rate)(x)
        outputs = layers.Dense(1)(x)

        self.model = keras.Model(inputs, outputs)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return self.model


def plot_training_history(history):
    fig = make_subplots(rows=2, cols=1)

    # Loss plot
    fig.add_trace(
        go.Scatter(y=history['loss'], name="Training Loss"),
        row=1, col=1
    )
    if 'val_loss' in history:
        fig.add_trace(
            go.Scatter(y=history['val_loss'], name="Validation Loss"),
            row=1, col=1
        )

    # Metrics plot
    for metric, values in history['metrics'].items():
        fig.add_trace(
            go.Scatter(y=values, name=metric),
            row=2, col=1
        )

    fig.update_layout(height=600, title_text="Training History")
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Metric Value", row=2, col=1)

    return fig


class SequenceValidator:
    """Comprehensive sequence validation and preparation for time series data."""

    def __init__(self, sequence_length, n_features=1):
        """
        Initialize validator with sequence parameters.

        Args:
            sequence_length (int): Length of input sequences
            n_features (int): Number of features in input data
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.scaler = None

    def validate_shape(self, X):
        """
        Validate input shape matches expected dimensions.

        Args:
            X: Input data to validate

        Raises:
            ValueError: If shape doesn't match expected dimensions
        """
        try:
            if len(X.shape) != 3:
                raise ValueError(
                    f"Expected 3D input (batch, sequence, features), got shape {X.shape}"
                )
            if X.shape[1] != self.sequence_length:
                raise ValueError(
                    f"Expected sequence length {self.sequence_length}, got {X.shape[1]}"
                )
            if X.shape[2] != self.n_features:
                raise ValueError(
                    f"Expected {self.n_features} features, got {X.shape[2]}"
                )
        except Exception as e:
            st.error(f"Shape validation error: {str(e)}")
            raise e

    def prepare_sequences(self, data, scale=True):
        """
        Prepare sequences from input data.

        Args:
            data: Input time series data
            scale: Whether to scale the data

        Returns:
            tuple: (X, y) sequences and targets
        """
        try:
            # Convert to numpy array if needed
            if isinstance(data, pd.Series):
                data = data.values
            elif isinstance(data, pd.DataFrame):
                data = data.values

            # Scale data if requested
            if scale:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()

            sequences = []
            targets = []

            # Create sequences
            for i in range(len(data) - self.sequence_length):
                seq = data[i:(i + self.sequence_length)]
                target = data[i + self.sequence_length]
                sequences.append(seq)
                targets.append(target)

            # Reshape sequences for LSTM/RNN input
            X = np.array(sequences)
            if len(X.shape) == 2:
                X = X.reshape((X.shape[0], X.shape[1], 1))
            y = np.array(targets)

            return X, y

        except Exception as e:
            st.error(f"Sequence preparation error: {str(e)}")
            return None, None

    def create_tf_dataset(self, X, y, batch_size=32, shuffle=True):
        """
        Create TensorFlow dataset from sequences.

        Args:
            X: Input sequences
            y: Target values
            batch_size: Size of batches
            shuffle: Whether to shuffle the data

        Returns:
            tf.data.Dataset: Batched dataset
        """
        try:
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
            if shuffle:
                dataset = dataset.shuffle(buffer_size=len(X))
            dataset = dataset.batch(batch_size)
            return dataset

        except Exception as e:
            st.error(f"Dataset creation error: {str(e)}")
            return None

    def split_data(self, X, y, train_size=0.8, val_size=0.1):
        """
        Split data into train/validation/test sets.

        Args:
            X: Input sequences
            y: Target values
            train_size: Proportion for training
            val_size: Proportion for validation

        Returns:
            tuple: (train_data, val_data, test_data)
        """
        try:
            n = len(X)
            train_end = int(n * train_size)
            val_end = train_end + int(n * val_size)

            train_data = (X[:train_end], y[:train_end])
            val_data = (X[train_end:val_end], y[train_end:val_end])
            test_data = (X[val_end:], y[val_end:])

            return train_data, val_data, test_data

        except Exception as e:
            st.error(f"Data splitting error: {str(e)}")
            return None, None, None

    def inverse_transform(self, data):
        """
        Inverse transform scaled data.

        Args:
            data: Scaled data to inverse transform

        Returns:
            array: Original scale data
        """
        try:
            if self.scaler is not None:
                if len(data.shape) == 1:
                    return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
                return self.scaler.inverse_transform(data)
            return data

        except Exception as e:
            st.error(f"Inverse transform error: {str(e)}")
            return None


# Continue in models/rnn_lstm.py after the main model classes

class ModelTrainer:
    """Comprehensive model trainer incorporating RNN/LSTM and confidence interval features."""

    def __init__(self, model_type, sequence_length=None, n_features=1):
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.validator = None
        self.monitor = TrainingMonitor()
        self.interval_predictor = None

        if model_type in ['LSTM', 'RNN', 'Stacked']:
            self.validator = SequenceValidator(sequence_length, n_features)

    def configure_optimizer(self):
        optimizer_type = st.selectbox(
            "Optimizer",
            ['adam', 'rmsprop', 'sgd']
        )
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=1e-6,
            max_value=1.0,
            value=0.001,
            format="%f"
        )

        if optimizer_type == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_type == 'rmsprop':
            return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            return tf.keras.optimizers.SGD(learning_rate=learning_rate)

    def train(self, data, validation_split=0.2):
        try:
            if self.model_type in ['LSTM', 'RNN', 'Stacked']:
                X, y = self.validator.prepare_sequences(data)
                train_data, val_data, test_data = self.validator.split_data(
                    X, y, train_size=0.8, val_size=validation_split
                )

                batch_size = st.select_slider(
                    "Batch Size",
                    options=[16, 32, 64, 128]
                )
                epochs = st.slider(
                    "Epochs",
                    min_value=10,
                    max_value=500,
                    value=100
                )

                train_ds = self.validator.create_tf_dataset(*train_data, batch_size)
                val_ds = self.validator.create_tf_dataset(*val_data, batch_size)

                optimizer = self.configure_optimizer()

                history = self.model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=epochs,
                    callbacks=[self.monitor],
                    optimizer=optimizer
                )

                self._setup_interval_predictor()
                return True

        except Exception as e:
            st.error(f"Training error: {str(e)}")
            return False

    def predict(self, data, return_intervals=True):
        try:
            X = self.validator.prepare_sequences(data)[0] if self.validator else data
            predictions = self.model.predict(X)

            if return_intervals and self.interval_predictor:
                intervals = self.interval_predictor.predict(X)
                return predictions, intervals

            return predictions

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None

    def _setup_interval_predictor(self):
        method = st.selectbox(
            "Confidence Interval Method",
            ['bootstrap', 'quantile', 'conformal']
        )
        confidence_level = st.slider(
            "Confidence Level",
            min_value=0.8,
            max_value=0.99,
            value=0.95
        )

        if method == 'bootstrap':
            self.interval_predictor = BootstrapPredictor(self.model, confidence_level)
        elif method == 'quantile':
            self.interval_predictor = QuantilePredictor(self.model, confidence_level)
        else:
            self.interval_predictor = ConformalPredictor(self.model, confidence_level)

    def plot_results(self, actual, predictions, intervals=None):
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=actual.index,
            y=actual,
            name='Actual',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=predictions.index,
            y=predictions,
            name='Predictions',
            line=dict(color='red', dash='dash')
        ))

        if intervals is not None:
            lower, upper = intervals
            fig.add_trace(go.Scatter(
                x=predictions.index.tolist() + predictions.index[::-1].tolist(),
                y=upper.tolist() + lower[::-1].tolist(),
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            ))

        fig.update_layout(
            title=f"{self.model_type} Predictions",
            xaxis_title="Time",
            yaxis_title="Value",
            template='plotly_white'
        )

        return fig


def plot_training_history(history):
    fig = make_subplots(rows=2, cols=1)

    fig.add_trace(
        go.Scatter(y=history['loss'], name="Training Loss"),
        row=1, col=1
    )
    if 'val_loss' in history:
        fig.add_trace(
            go.Scatter(y=history['val_loss'], name="Validation Loss"),
            row=1, col=1
        )

    for metric, values in history['metrics'].items():
        fig.add_trace(
            go.Scatter(y=values, name=metric),
            row=2, col=1
        )

    fig.update_layout(height=600, title_text="Training History")
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Metric Value", row=2, col=1)

    return fig

class BaseIntervalPredictor:
    """Base class for prediction interval calculation."""

    def __init__(self, model, confidence_level=0.95):
        self.model = model
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

class BootstrapPredictor(BaseIntervalPredictor):
    """Bootstrap method for prediction intervals."""

    def __init__(self, model, confidence_level=0.95, n_samples=200):
        super().__init__(model, confidence_level)
        self.n_samples = n_samples
        self.bootstrap_predictions = None

    def fit(self, X, y):
        """Fit bootstrap samples."""
        try:
            n = len(X)
            self.bootstrap_predictions = []

            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i in range(self.n_samples):
                # Random sampling with replacement
                indices = np.random.choice(n, size=n, replace=True)
                X_boot, y_boot = X[indices], y[indices]

                # Train model on bootstrap sample
                self.model.fit(X_boot, y_boot)
                self.bootstrap_predictions.append(self.model.predict(X))

                # Update progress
                progress = (i + 1) / self.n_samples
                progress_bar.progress(progress)
                status_text.text(f"Bootstrap sample {i + 1}/{self.n_samples}")

            return True

        except Exception as e:
            st.error(f"Bootstrap fitting error: {str(e)}")
            return False

    def predict(self, X):
        """Generate predictions with bootstrap intervals."""
        try:
            predictions = np.array(self.bootstrap_predictions)
            mean_pred = np.mean(predictions, axis=0)
            lower = np.percentile(predictions, (self.alpha / 2) * 100, axis=0)
            upper = np.percentile(predictions, (1 - self.alpha / 2) * 100, axis=0)

            return mean_pred, (lower, upper)

        except Exception as e:
            st.error(f"Bootstrap prediction error: {str(e)}")
            return None

class QuantilePredictor(BaseIntervalPredictor):
    """Quantile regression for prediction intervals."""

    def __init__(self, model, confidence_level=0.95):
        super().__init__(model, confidence_level)
        self.lower_model = None
        self.upper_model = None

    def fit(self, X, y):
        """Fit quantile regression models."""
        try:
            from sklearn.ensemble import GradientBoostingRegressor

            # Create lower and upper quantile models
            self.lower_model = GradientBoostingRegressor(
                loss='quantile',
                alpha=self.alpha / 2
            )
            self.upper_model = GradientBoostingRegressor(
                loss='quantile',
                alpha=1 - self.alpha / 2
            )

            # Fit models
            self.lower_model.fit(X, y)
            self.upper_model.fit(X, y)

            return True

        except Exception as e:
            st.error(f"Quantile fitting error: {str(e)}")
            return False

    def predict(self, X):
        """Generate predictions with quantile intervals."""
        try:
            predictions = self.model.predict(X)
            lower = self.lower_model.predict(X)
            upper = self.upper_model.predict(X)

            return predictions, (lower, upper)

        except Exception as e:
            st.error(f"Quantile prediction error: {str(e)}")
            return None

class ConformalPredictor(BaseIntervalPredictor):
    """Conformal prediction for prediction intervals."""

    def __init__(self, model, confidence_level=0.95):
        super().__init__(model, confidence_level)
        self.residuals = None
        self.conformity_scores = None

    def fit(self, X, y):
        """Fit conformal predictor."""
        try:
            # Generate predictions on calibration set
            predictions = self.model.predict(X)

            # Calculate residuals and conformity scores
            self.residuals = np.abs(y - predictions)
            self.conformity_scores = np.sort(self.residuals)

            return True

        except Exception as e:
            st.error(f"Conformal fitting error: {str(e)}")
            return False

    def predict(self, X):
        """Generate predictions with conformal intervals."""
        try:
            predictions = self.model.predict(X)

            # Calculate prediction interval width
            n = len(self.conformity_scores)
            index = int(np.ceil((n + 1) * (1 - self.alpha)))
            width = self.conformity_scores[index - 1]

            lower = predictions - width
            upper = predictions + width

            return predictions, (lower, upper)

        except Exception as e:
            st.error(f"Conformal prediction error: {str(e)}")
            return None