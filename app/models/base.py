"""Base model class for time series forecasting."""

from abc import ABC, abstractmethod
import streamlit as st


class TimeSeriesModel(ABC):
    """Abstract base class for all time series models."""

    def __init__(self, name):
        self.name = name
        self.model = None
        self.train_params = {}
        self.build_params = {}

    @abstractmethod
    def build(self, **kwargs):
        """Build the model architecture."""
        pass

    # In base.py, update the train method in TimeSeriesModel class:

    def train(self, train_data, val_data=None, **kwargs):
        """
        Train the model with appropriate parameters based on model type.
        """
        try:
            # For deep learning models (using Keras)
            if hasattr(self.model, 'fit'):
                epochs = self.train_params.get('epochs', 100)
                batch_size = self.train_params.get('batch_size', 32)
                callbacks = self.train_params.get('callbacks', None)
                validation_data = val_data if val_data is not None else None

                return self.model.fit(
                    train_data,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=validation_data,
                    callbacks=callbacks
                )

            # For traditional ML models (using sklearn-like API)
            elif hasattr(self.model, 'fit'):
                return self.model.fit(train_data[0], train_data[1])

            # For custom implementations
            else:
                return self._custom_train(train_data, val_data)

        except Exception as e:
            st.error(f"Error in model training: {str(e)}")
            return None

    def _custom_train(self, train_data, val_data=None, **kwargs):
        """Custom training implementation for specific model types."""
        raise NotImplementedError("Custom training not implemented for this model type")

    @abstractmethod
    def evaluate(self, test_data):
        """Evaluate the model."""
        pass

    @abstractmethod
    def save(self, path):
        """Save the model."""
        pass

    @abstractmethod
    def load(self, path):
        """Load the model."""
        pass

    def predict(self, X, **kwargs):
        """
        Make predictions with confidence intervals if supported.

        Args:
            X: Input data
            **kwargs: Additional prediction parameters
                - return_intervals: Whether to return confidence intervals
                - confidence_level: Confidence level for intervals
                - n_iterations: Number of bootstrap iterations
        """
        try:
            if hasattr(self.model, 'predict'):
                predictions = self.model.predict(X)

                # Calculate confidence intervals if requested
                if kwargs.get('return_intervals', False):
                    conf_level = kwargs.get('confidence_level', 0.95)
                    n_iter = kwargs.get('n_iterations', 100)
                    lower, upper = self._calculate_intervals(X, predictions, conf_level, n_iter)
                    return predictions, lower, upper

                return predictions
            else:
                raise NotImplementedError("Predict method not implemented")

        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            return None

    def _calculate_intervals(self, X, predictions, confidence_level=0.95, n_iterations=100):
        """Calculate confidence intervals using bootstrapping."""
        try:
            if hasattr(self, '_bootstrap_predict'):
                return self._bootstrap_predict(X, confidence_level, n_iterations)
            else:
                return None, None
        except Exception as e:
            st.error(f"Error calculating intervals: {str(e)}")
            return None, None