"""Deep learning model implementations."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .base import TimeSeriesModel
import streamlit as st
import numpy as np

from .confidence_intervals import generate_prediction_intervals

class DeepLearningModel(TimeSeriesModel):
    """Base class for deep learning models."""

    def __init__(self, name, sequence_length, n_features=1):
        super().__init__(name)
        self.sequence_length = sequence_length
        self.n_features = n_features

    def build(self, **kwargs):
        """Base build method - should be overridden by subclasses."""
        raise NotImplementedError

    def save(self, path):
        """Save the Keras model."""
        self.model.save(path)

    def load(self, path):
        """Load the Keras model."""
        self.model = keras.models.load_model(path)

    def train(self, train_data, val_data=None):
        """Train the model using parameters from train_params."""
        if not hasattr(self, 'train_params'):
            self.train_params = {}

        batch_size = self.train_params.get('batch_size', 32)
        epochs = self.train_params.get('epochs', 50)
        callbacks = self.train_params.get('callbacks', None)

        return self.model.fit(
            train_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            callbacks=callbacks
        )

    def evaluate(self, test_data):
        """Evaluate the model."""
        return self.model.evaluate(test_data)


class SequentialModel(DeepLearningModel):
    """Sequential model with configurable layers."""

    def build(self, **kwargs):
        """Build Sequential model with configured layers."""
        try:
            inputs = keras.Input(shape=(self.sequence_length, self.n_features))
            x = inputs

            # Extract layer parameters from kwargs
            layer_params = {}
            for key in list(kwargs.keys()):
                if key.startswith('layer_'):
                    layer_params[key] = kwargs.pop(key)

            # Sort layers by their index
            sorted_layers = sorted(layer_params.items(), key=lambda x: int(x[0].split('_')[1]))

            # Process each layer
            for idx, (_, layer_config) in enumerate(sorted_layers):
                layer_type = layer_config['type'].lower()
                units = layer_config['units']
                activation = layer_config['activation']
                dropout = layer_config['dropout']

                # Add the appropriate layer type
                if layer_type == 'dense':
                    x = layers.Dense(units, activation=activation)(x)
                elif layer_type == 'lstm':
                    return_sequences = idx < len(sorted_layers) - 1  # True if not last layer
                    x = layers.LSTM(units, activation=activation,
                                  return_sequences=return_sequences)(x)
                elif layer_type == 'gru':
                    return_sequences = idx < len(sorted_layers) - 1  # True if not last layer
                    x = layers.GRU(units, activation=activation,
                                 return_sequences=return_sequences)(x)
                elif layer_type == 'simple_rnn':
                    return_sequences = idx < len(sorted_layers) - 1  # True if not last layer
                    x = layers.SimpleRNN(units, activation=activation,
                                       return_sequences=return_sequences)(x)

                # Add dropout if specified
                if dropout > 0:
                    x = layers.Dropout(dropout)(x)

            # Add final dense layer for prediction
            outputs = layers.Dense(1)(x)
            self.model = keras.Model(inputs, outputs)

            # Configure optimizer
            optimizer = kwargs.get('optimizer', 'adam')
            if isinstance(optimizer, str):
                learning_rate = kwargs.get('learning_rate', 0.001)
                optimizer = keras.optimizers.get(optimizer)
                optimizer.learning_rate = learning_rate

            # Compile model
            self.model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae']
            )

            st.write(f"Model input shape: {inputs.shape}")  # Debug info
            return self.model

        except Exception as e:
            st.error(f"Error building Sequential model: {str(e)}")
            st.write("Debug info:")
            st.write(f"Sequence length: {self.sequence_length}")
            st.write(f"Features: {self.n_features}")
            st.write(f"Layer params: {layer_params}")
            return None


class SimpleRNNModel(SequentialModel):
    """Simple RNN model implementation."""

    def build(self, units=64, dropout_rate=0.1, **kwargs):
        """Build Simple RNN model using sequential build method."""
        return super().build(
            layer_0={
                'type': 'simple_rnn',
                'units': units,
                'activation': 'tanh',
                'dropout': dropout_rate
            },
            **kwargs
        )


class LSTMModel(SequentialModel):
    """LSTM model implementation."""

    def build(self, units=64, dropout_rate=0.1, **kwargs):
        """Build LSTM model using sequential build method."""
        return super().build(
            layer_0={
                'type': 'lstm',
                'units': units,
                'activation': 'tanh',
                'dropout': dropout_rate
            },
            **kwargs
        )


class StackedModel(SequentialModel):
    """Stacked LSTM+RNN model implementation."""

    def build(self, lstm_units=128, rnn_units=64, dropout_rate=0.1, **kwargs):
        """Build stacked model using sequential build method."""
        return super().build(
            layer_0={
                'type': 'lstm',
                'units': lstm_units,
                'activation': 'tanh',
                'dropout': dropout_rate
            },
            layer_1={
                'type': 'simple_rnn',
                'units': rnn_units,
                'activation': 'tanh',
                'dropout': dropout_rate
            },
            **kwargs
        )
        
    def predict(self, X, show_ci=False, ci_method="bootstrapping", prediction_type="one-step", **kwargs):
        """Generate predictions and optionally return confidence intervals."""
        try:
            # Get model predictions
            predictions = self.model.predict(X)

            if show_ci:
                # Call the generate_prediction_intervals function to get confidence intervals
                ci_results = generate_prediction_intervals(
                    df=kwargs['df'],  # Pass your time series dataframe
                    target_variable=kwargs['target_column'],  # Target column
                    model=self.model,  # Model instance
                    method=ci_method,  # User selected CI method
                    prediction_type=prediction_type,  # User selected prediction type
                    alpha=kwargs.get('alpha', 0.05)  # Confidence level (default: 0.05)
                )
                return predictions.squeeze(), ci_results
            else:
                return predictions
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            return None
    
    def predict(self, X):
        """Generate predictions."""
        try:
            predictions = self.model.predict(X)
            return predictions.squeeze()
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            return None

    def evaluate(self, test_data):
        """Evaluate model on test data."""
        try:
            if isinstance(test_data, tuple):
                X_test, y_test = test_data
            else:
                X_test, y_test = test_data.data, test_data.targets

            predictions = self.predict(X_test)

            # Calculate metrics
            mae = tf.keras.metrics.mean_absolute_error(y_test, predictions)
            mse = tf.keras.metrics.mean_squared_error(y_test, predictions)
            rmse = tf.math.sqrt(mse)

            return {
                'MAE': float(mae),
                'RMSE': float(rmse),
                'MSE': float(mse)
            }
        except Exception as e:
            st.error(f"Error in evaluation: {str(e)}")
            return None

    def summary(self):
        """Print model summary."""
        if self.model is not None:
            return self.model.summary()
        return "Model not built yet."


def create_deep_learning_model(model_type: str, sequence_length: int, n_features: int = 1, **kwargs):
    """Factory function to create deep learning models."""
    if model_type == 'Simple RNN':
        return SimpleRNNModel(name='Simple RNN', sequence_length=sequence_length, n_features=n_features)
    elif model_type == 'LSTM':
        return LSTMModel(name='LSTM', sequence_length=sequence_length, n_features=n_features)
    elif model_type == 'Stacked LSTM+RNN':
        return StackedModel(name='Stacked LSTM+RNN', sequence_length=sequence_length, n_features=n_features)
    elif model_type == 'Sequential':
        return SequentialModel(name='Sequential', sequence_length=sequence_length, n_features=n_features)
    else:
        raise ValueError(f"Unknown model type: {model_type}")