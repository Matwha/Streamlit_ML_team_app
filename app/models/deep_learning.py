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