"""Traditional time series model implementations."""

import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .base import TimeSeriesModel
import streamlit as st


class ARIMAModel(TimeSeriesModel):
    """ARIMA model implementation."""

    def __init__(self, name="ARIMA"):
        super().__init__(name)
        self.order = None
        self.fitted_model = None

    def build(self, order=None):
        """
        Build ARIMA model with specified or auto-determined order.

        Args:
            order (tuple): ARIMA order (p,d,q)
        """
        self.order = order
        return self

    def auto_order(self, train_data, seasonal=False):
        """
        Automatically determine ARIMA order using pmdarima.

        Args:
            train_data (array-like): Training data
            seasonal (bool): Whether to include seasonal components

        Returns:
            tuple: ARIMA order
        """
        try:
            auto_arima = pm.auto_arima(
                train_data,
                seasonal=seasonal,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                max_order=None,
                trace=False
            )
            return auto_arima.order

        except Exception as e:
            st.error(f"Error in auto order determination: {str(e)}")
            return (1, 1, 1)  # default fallback order

    def train(self, train_data, val_data=None, **kwargs):
        """
        Train ARIMA model.

        Args:
            train_data (array-like): Training data
            val_data (array-like): Validation data (not used for ARIMA)
            **kwargs: Additional arguments

        Returns:
            self: Trained model
        """
        try:
            if self.order is None:
                self.order = self.auto_order(train_data)

            self.fitted_model = ARIMA(
                train_data,
                order=self.order
            ).fit()

            return self

        except Exception as e:
            st.error(f"Error training ARIMA model: {str(e)}")
            return None

    def predict(self, steps=1):
        """
        Generate predictions.

        Args:
            steps (int): Number of steps to forecast

        Returns:
            array-like: Predictions
        """
        try:
            return self.fitted_model.forecast(steps=steps)
        except Exception as e:
            st.error(f"Error in ARIMA prediction: {str(e)}")
            return None

    def evaluate(self, test_data):
        """
        Evaluate model on test data.

        Args:
            test_data (array-like): Test data

        Returns:
            dict: Dictionary of evaluation metrics
        """
        try:
            predictions = self.predict(steps=len(test_data))
            mae = np.mean(np.abs(predictions - test_data))
            rmse = np.sqrt(np.mean((predictions - test_data) ** 2))
            mape = np.mean(np.abs((predictions - test_data) / test_data)) * 100

            return {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape
            }

        except Exception as e:
            st.error(f"Error evaluating ARIMA model: {str(e)}")
            return None

    def save(self, path):
        """
        Save model to file.

        Args:
            path (str): Path to save model
        """
        if self.fitted_model is not None:
            try:
                self.fitted_model.save(path)
            except Exception as e:
                st.error(f"Error saving ARIMA model: {str(e)}")

    def load(self, path):
        """
        Load model from file.

        Args:
            path (str): Path to load model from
        """
        try:
            self.fitted_model = ARIMA.load(path)
        except Exception as e:
            st.error(f"Error loading ARIMA model: {str(e)}")


class SARIMAModel(TimeSeriesModel):
    """SARIMA model implementation."""

    def __init__(self, name="SARIMA"):
        super().__init__(name)
        self.order = None
        self.seasonal_order = None
        self.fitted_model = None

    def build(self, order=None, seasonal_order=None):
        """
        Build SARIMA model with specified or auto-determined orders.

        Args:
            order (tuple): ARIMA order (p,d,q)
            seasonal_order (tuple): Seasonal order (P,D,Q,s)
        """
        self.order = order
        self.seasonal_order = seasonal_order
        return self

    def auto_order(self, train_data):
        """
        Automatically determine SARIMA orders using pmdarima.

        Args:
            train_data (array-like): Training data

        Returns:
            tuple: (order, seasonal_order)
        """
        try:
            auto_arima = pm.auto_arima(
                train_data,
                seasonal=True,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                max_order=None,
                trace=False
            )
            return auto_arima.order, auto_arima.seasonal_order

        except Exception as e:
            st.error(f"Error in auto order determination: {str(e)}")
            return (1, 1, 1), (1, 1, 1, 12)  # default fallback orders

    def train(self, train_data, val_data=None, **kwargs):
        """
        Train SARIMA model.

        Args:
            train_data (array-like): Training data
            val_data (array-like): Validation data (not used for SARIMA)
            **kwargs: Additional arguments

        Returns:
            self: Trained model
        """
        try:
            if self.order is None or self.seasonal_order is None:
                self.order, self.seasonal_order = self.auto_order(train_data)

            self.fitted_model = SARIMAX(
                train_data,
                order=self.order,
                seasonal_order=self.seasonal_order
            ).fit(disp=False)

            return self

        except Exception as e:
            st.error(f"Error training SARIMA model: {str(e)}")
            return None

    def predict(self, steps=1):
        """
        Generate predictions.

        Args:
            steps (int): Number of steps to forecast

        Returns:
            array-like: Predictions
        """
        try:
            return self.fitted_model.forecast(steps=steps)
        except Exception as e:
            st.error(f"Error in SARIMA prediction: {str(e)}")
            return None

    def evaluate(self, test_data):
        """
        Evaluate model on test data.

        Args:
            test_data (array-like): Test data

        Returns:
            dict: Dictionary of evaluation metrics
        """
        try:
            predictions = self.predict(steps=len(test_data))
            mae = np.mean(np.abs(predictions - test_data))
            rmse = np.sqrt(np.mean((predictions - test_data) ** 2))
            mape = np.mean(np.abs((predictions - test_data) / test_data)) * 100

            return {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape
            }

        except Exception as e:
            st.error(f"Error evaluating SARIMA model: {str(e)}")
            return None

    def save(self, path):
        """Save model to file."""
        if self.fitted_model is not None:
            try:
                self.fitted_model.save(path)
            except Exception as e:
                st.error(f"Error saving SARIMA model: {str(e)}")

    def load(self, path):
        """Load model from file."""
        try:
            self.fitted_model = SARIMAX.load(path)
        except Exception as e:
            st.error(f"Error loading SARIMA model: {str(e)}")


def create_traditional_model(model_type, **kwargs):
    """
    Factory function to create traditional models.

    Args:
        model_type (str): Type of model ('ARIMA' or 'SARIMA')
        **kwargs: Additional arguments for model construction

    Returns:
        TimeSeriesModel: Instance of requested model
    """
    if model_type.upper() == 'ARIMA':
        return ARIMAModel(**kwargs)
    elif model_type.upper() == 'SARIMA':
        return SARIMAModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")