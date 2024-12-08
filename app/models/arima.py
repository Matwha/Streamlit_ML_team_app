# models/arima.py
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go

class ARIMAModel:
    def __init__(self, order=(1,1,1)):
        self.order = order
        self.model = None
        self.fitted_model = None
        self.training_summary = {}

    def train(self, data, **kwargs):
        try:
            self.model = ARIMA(data, order=self.order)
            self.fitted_model = self.model.fit()
            self._update_training_summary()
            return self.fitted_model
        except Exception as e:
            st.error(f"ARIMA training error: {str(e)}")
            return None

    def predict(self, steps=1, return_conf_int=True, alpha=0.05):
        try:
            forecast = self.fitted_model.forecast(steps=steps)
            if return_conf_int:
                conf_int = self.fitted_model.get_forecast(steps=steps).conf_int(alpha=alpha)
                return forecast, conf_int
            return forecast
        except Exception as e:
            st.error(f"ARIMA prediction error: {str(e)}")
            return None

    def _update_training_summary(self):
        self.training_summary = {
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'residuals': self.fitted_model.resid
        }

class SARIMAModel:
    def __init__(self, order=(1,1,1), seasonal_order=(1,1,1,12)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        self.training_summary = {}

    def train(self, data, **kwargs):
        try:
            self.model = SARIMAX(data,
                               order=self.order,
                               seasonal_order=self.seasonal_order)
            self.fitted_model = self.model.fit()
            self._update_training_summary()
            return self.fitted_model
        except Exception as e:
            st.error(f"SARIMA training error: {str(e)}")
            return None

    def predict(self, steps=1, return_conf_int=True, alpha=0.05):
        try:
            forecast = self.fitted_model.forecast(steps=steps)
            if return_conf_int:
                conf_int = self.fitted_model.get_forecast(steps=steps).conf_int(alpha=alpha)
                return forecast, conf_int
            return forecast
        except Exception as e:
            st.error(f"SARIMA prediction error: {str(e)}")
            return None

    def _update_training_summary(self):
        self.training_summary = {
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'residuals': self.fitted_model.resid
        }

def plot_results(actual, predictions, conf_int=None, title="Forecast Results"):
    fig = go.Figure()

    # Actual values
    fig.add_trace(go.Scatter(
        x=actual.index,
        y=actual,
        mode='lines',
        name='Actual',
        line=dict(color='blue')
    ))

    # Predictions
    fig.add_trace(go.Scatter(
        x=predictions.index,
        y=predictions,
        mode='lines',
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))

    # Confidence intervals
    if conf_int is not None:
        fig.add_trace(go.Scatter(
            x=predictions.index.tolist() + predictions.index[::-1].tolist(),
            y=conf_int[:, 1].tolist() + conf_int[:, 0][::-1].tolist(),
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Value',
        template='plotly_white'
    )

    return fig