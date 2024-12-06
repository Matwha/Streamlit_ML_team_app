"""Deep learning model implementations."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .base import TimeSeriesModel
import streamlit as st


class DeepLearningModel(TimeSeriesModel):
    """Base class for deep learning models."""

    def __init__(self, name, sequence_length, n_features=1):
        super().__init__(name)
        self.sequence_length = sequence_length
        self.n_features = n_features

    def save(self, path):
        """Save the Keras model."""
        self.model.save(path)

    def load(self, path):
        """Load the Keras model."""
        self.model = keras.models.load_model(path)

    def train(self, train_data, val_data=None, epochs=10, callbacks=None):
        """Train the model."""
        return self.model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
            callbacks=callbacks
        )

    def evaluate(self, test_data):
        """Evaluate the model."""
        return self.model.evaluate(test_data)


class SimpleRNNModel(DeepLearningModel):
    """Simple RNN model implementation."""

    def build(self, units=64, dropout_rate=0.1):
        """Build Simple RNN model."""
        inputs = keras.Input(shape=(self.sequence_length, self.n_features))
        x = layers.SimpleRNN(units, dropout=dropout_rate)(inputs)
        outputs = layers.Dense(1)(x)
        self.model = keras.Model(inputs, outputs)
        return self.model


class LSTMModel(DeepLearningModel):
    """LSTM model implementation."""

    def build(self, units=64, dropout_rate=0.1):
        """Build LSTM model."""
        inputs = keras.Input(shape=(self.sequence_length, self.n_features))
        x = layers.LSTM(units, dropout=dropout_rate)(inputs)
        outputs = layers.Dense(1)(x)
        self.model = keras.Model(inputs, outputs)
        return self.model


class StackedModel(DeepLearningModel):
    """Stacked LSTM+RNN model implementation."""

    def build(self, lstm_units=128, rnn_units=64, dropout_rate=0.1):
        """Build stacked LSTM+RNN model."""
        inputs = keras.Input(shape=(self.sequence_length, self.n_features))
        x = layers.LSTM(lstm_units, dropout=dropout_rate, return_sequences=True)(inputs)
        x = layers.SimpleRNN(rnn_units, dropout=dropout_rate)(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1)(x)
        self.model = keras.Model(inputs, outputs)
        return self.model


class SequentialModel(DeepLearningModel):
    """Sequential model implementation with customizable layers."""

    def __init__(self, name="Sequential", sequence_length=10, n_features=1):
        super().__init__(name, sequence_length, n_features)
        self.layers_config = []

    def add_layer(self, layer_type, units, activation='relu', dropout=0.0):
        """Add a layer to the model configuration."""
        self.layers_config.append({
            'type': layer_type,
            'units': units,
            'activation': activation,
            'dropout': dropout
        })

    def build(self, learning_rate=0.001):
        """Build Sequential model with configured layers."""
        inputs = keras.Input(shape=(self.sequence_length, self.n_features))
        x = inputs

        for layer_config in self.layers_config:
            if layer_config['type'].lower() == 'dense':
                x = layers.Dense(
                    layer_config['units'],
                    activation=layer_config['activation']
                )(x)
            elif layer_config['type'].lower() == 'lstm':
                x = layers.LSTM(
                    layer_config['units'],
                    activation=layer_config['activation'],
                    return_sequences=True
                )(x)
            elif layer_config['type'].lower() == 'gru':
                x = layers.GRU(
                    layer_config['units'],
                    activation=layer_config['activation'],
                    return_sequences=True
                )(x)

            if layer_config['dropout'] > 0:
                x = layers.Dropout(layer_config['dropout'])(x)

        # Add final dense layer for prediction
        outputs = layers.Dense(1)(x)

        self.model = keras.Model(inputs, outputs)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )

        return self.model

    def summary(self):
        """Print model summary."""
        return self.model.summary()