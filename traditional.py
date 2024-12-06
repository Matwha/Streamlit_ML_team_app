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
        self.test_data = None

    def build(self, **kwargs):
        """
        Build ARIMA model with specified or auto-determined order.

        Args:
            p (int): AR order
            d (int): Differencing order
            q (int): MA order
        """
        try:
            # Extract order parameters from kwargs
            p = kwargs.get('p', 1)
            d = kwargs.get('d', 1)
            q = kwargs.get('q', 1)
            
            self.order = (p, d, q)
            return self

        except Exception as e:
            st.error(f"Error building ARIMA model: {str(e)}")
            return None

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

    def predict(self, steps=1, bootstrapping=False, n_bootstraps=100, alpha=0.05):
        """Generate predictions and optionally return bootstrapped confidence intervals."""
        try:
            forecast = self.fitted_model.forecast(steps=steps)
            
            if bootstrapping:
                # Generate bootstrapped predictions
                residuals = self.fitted_model.resid
                lower_bounds = []
                upper_bounds = []

                for _ in range(n_bootstraps):
                    # Resample residuals with replacement
                    resampled_residuals = np.random.choice(residuals, size=len(residuals), replace=True)
                    # Add the resampled residuals to the forecast
                    bootstrap_forecast = forecast + resampled_residuals[:steps]
                    lower_bounds.append(np.percentile(bootstrap_forecast, 100 * alpha / 2))
                    upper_bounds.append(np.percentile(bootstrap_forecast, 100 * (1 - alpha / 2)))
                
                # Calculate the confidence intervals from the bootstrapped forecasts
                lower_bound = np.percentile(lower_bounds, 50)  # Median of lower bounds
                upper_bound = np.percentile(upper_bounds, 50)  # Median of upper bounds

                return forecast, lower_bound, upper_bound

            return forecast
        except Exception as e:
            st.error(f"Error in ARIMA prediction: {str(e)}")
            return None

    def calculate_prediction_intervals(self, alpha=0.95):
        """
        Calculate prediction intervals for forecasts.

        Args:
            alpha (float): Confidence level (between 0 and 1)

        Returns:
            tuple: (lower bound array, upper bound array)
        """
        try:
            if self.fitted_model is None:
                raise ValueError("Model has not been trained yet")
            
            forecast_obj = self.fitted_model.get_forecast(steps=len(self.test_data))
            conf_int = forecast_obj.conf_int(alpha=alpha)
            return conf_int[:, 0], conf_int[:, 1]  # Lower and upper bounds
            
        except Exception as e:
            st.error(f"Error calculating prediction intervals: {str(e)}")
            return None, None

    def evaluate(self, test_data):
        """
        Evaluate model on test data.

        Args:
            test_data (array-like): Test data

        Returns:
            dict: Dictionary of evaluation metrics
        """
        try:
            self.test_data = test_data
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

    def predict(self, steps=1, bootstrapping=False, n_bootstraps=100, alpha=0.05):
        """Generate predictions and optionally return bootstrapped confidence intervals."""
        try:
            forecast = self.fitted_model.forecast(steps=steps)

            if bootstrapping:
                # Generate bootstrapped predictions
                residuals = self.fitted_model.resid
                lower_bounds = []
                upper_bounds = []

                for _ in range(n_bootstraps):
                    # Resample residuals with replacement
                    resampled_residuals = np.random.choice(residuals, size=len(residuals), replace=True)
                    # Add the resampled residuals to the forecast
                    bootstrap_forecast = forecast + resampled_residuals[:steps]
                    lower_bounds.append(np.percentile(bootstrap_forecast, 100 * alpha / 2))
                    upper_bounds.append(np.percentile(bootstrap_forecast, 100 * (1 - alpha / 2)))
                
                # Calculate the confidence intervals from the bootstrapped forecasts
                lower_bound = np.percentile(lower_bounds, 50)  # Median of lower bounds
                upper_bound = np.percentile(upper_bounds, 50)  # Median of upper bounds

                return forecast, lower_bound, upper_bound

            return forecast
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