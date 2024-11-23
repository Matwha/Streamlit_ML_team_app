import streamlit as st
import pandas as pd
import numpy as np
from pycaret.time_series import *
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings


warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Economic Time Series Analysis", layout="wide")


def try_multiple_formats(value):
    """Try multiple date formats for parsing"""
    formats = ['%m/%d/%Y', '%Y-%m-%d', '%Y-%b', '%y-%b', '%Y-%m']
    for fmt in formats:
        try:
            parsed = pd.to_datetime(value, format=fmt, errors='coerce')
            if parsed is not pd.NaT:
                return parsed
        except ValueError:
            continue
    return pd.NaT


def load_data(uploaded_file):
    """Load and preprocess uploaded data"""
    if uploaded_file is not None:
        try:
            # Read the CSV file
            data = pd.read_csv(uploaded_file)

            # Convert the first column to datetime
            try:
                data.iloc[:, 0] = data.iloc[:, 0].apply(try_multiple_formats)
                data.set_index(data.columns[0], inplace=True)
                data.sort_index(inplace=True)

                # Convert index to quarterly period
                data.index = pd.PeriodIndex(data.index, freq='Q')

                # Ensure all numeric columns are float type
                for col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')

                return data
            except Exception as e:
                st.warning(f"Date conversion notice: {str(e)}")
                return None
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    return None


def get_model_parameters(model_type):
    """Get comprehensive model parameters based on selected model type"""
    params = {}

    with st.sidebar:
        st.subheader("Model Configuration")

        # Common parameters for all models
        params['train_size'] = st.slider(
            "Training Data Split (%)",
            50, 95, 80,
            help="Percentage of data to use for training"
        ) / 100

        params['forecast_horizon'] = st.slider(
            "Forecast Horizon (Quarters)",
            1, 20, 4,
            help="Number of quarters to forecast ahead"
        )

        if model_type == 'ets':
            st.subheader("ETS Parameters")
            params.update({
                'error': st.selectbox(
                    "Error Type",
                    options=['add', 'mul'],
                    help="Additive or multiplicative error"
                ),
                'trend': st.selectbox(
                    "Trend Type",
                    options=['add', 'mul', None],
                    help="Type of trend component"
                ),
                'seasonal': st.selectbox(
                    "Seasonal Type",
                    options=['add', 'mul', None],
                    help="Type of seasonality"
                ),
                'damped_trend': st.checkbox(
                    "Damped Trend",
                    value=False,
                    help="Apply damping to trend"
                ),
                'seasonal_periods': st.number_input(
                    "Seasonal Periods",
                    min_value=1,
                    value=4,
                    help="Number of periods in one seasonal cycle"
                )
            })

        elif model_type == 'arima':
            st.subheader("ARIMA Parameters")
            col1, col2 = st.columns(2)

            with col1:
                params.update({
                    'p': st.slider("AR Order (p)", 0, 5, 2),
                    'd': st.slider("Difference Order (d)", 0, 2, 1),
                    'q': st.slider("MA Order (q)", 0, 5, 2),
                })

            with col2:
                params.update({
                    'P': st.slider("Seasonal AR (P)", 0, 2, 1),
                    'D': st.slider("Seasonal Difference (D)", 0, 1, 1),
                    'Q': st.slider("Seasonal MA (Q)", 0, 2, 1),
                })

            params['seasonal'] = st.checkbox("Include Seasonality", value=True)
            if params['seasonal']:
                params['seasonal_periods'] = st.number_input(
                    "Seasonal Periods",
                    min_value=1,
                    value=4,
                    help="Number of periods in seasonal cycle"
                )

        elif model_type == 'prophet':
            st.subheader("Prophet Parameters")

            params.update({
                'growth': st.selectbox(
                    "Growth Model",
                    options=['linear', 'logistic'],
                    help="Type of growth trend"
                ),
                'seasonality_mode': st.selectbox(
                    "Seasonality Mode",
                    options=['additive', 'multiplicative'],
                    help="Type of seasonality"
                ),
                'seasonality_prior_scale': st.slider(
                    "Seasonality Prior Scale",
                    min_value=0.01,
                    max_value=10.0,
                    value=10.0,
                    help="Strength of seasonality"
                ),
                'changepoint_prior_scale': st.slider(
                    "Changepoint Prior Scale",
                    min_value=0.001,
                    max_value=0.5,
                    value=0.05,
                    help="Flexibility of trend"
                ),
                'changepoint_range': st.slider(
                    "Changepoint Range",
                    min_value=0.1,
                    max_value=0.9,
                    value=0.8,
                    help="Proportion of history where trend changes are considered"
                )
            })

        # Advanced Options
        if st.checkbox("Show Advanced Options", False):
            st.subheader("Advanced Settings")
            params['confidence_level'] = st.slider(
                "Confidence Level",
                min_value=0.8,
                max_value=0.99,
                value=0.9,
                step=0.01,
                help="Confidence level for prediction intervals"
            )

            params['cross_validation'] = st.checkbox(
                "Enable Cross Validation",
                value=False,
                help="Perform time series cross validation"
            )

            if params['cross_validation']:
                params['cv_folds'] = st.slider(
                    "Number of CV Folds",
                    min_value=2,
                    max_value=10,
                    value=5
                )

    return params


def calculate_metrics(y_true, y_pred):
    """Calculate forecast accuracy metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }


def plot_forecast(data, predictions, title, metrics=None):
    """Create interactive forecast plot"""
    fig = go.Figure()

    # Plot actual data
    fig.add_trace(go.Scatter(
        x=data.index.astype(str),
        y=data.values,
        name='Actual',
        line=dict(color='blue')
    ))

    # Plot predictions
    fig.add_trace(go.Scatter(
        x=predictions.index.astype(str),
        y=predictions['y_pred'],
        name='Forecast',
        line=dict(color='red')
    ))

    # Add confidence intervals if available
    if 'y_pred_lower' in predictions.columns:
        fig.add_trace(go.Scatter(
            x=predictions.index.astype(str),
            y=predictions['y_pred_lower'],
            fill=None,
            mode='lines',
            line=dict(color='rgba(255,0,0,0)'),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=predictions.index.astype(str),
            y=predictions['y_pred_upper'],
            fill='tonexty',
            mode='lines',
            line=dict(color='rgba(255,0,0,0)'),
            name='95% Confidence Interval',
            fillcolor='rgba(255,0,0,0.2)'
        ))

    # Add metrics annotation if provided
    if metrics:
        metrics_text = '<br>'.join([f"{k}: {v:.2f}" for k, v in metrics.items()])
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            text=metrics_text,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified',
        showlegend=True,
        template='plotly_white'
    )

    return fig


def main():
    # Rest of the main function implementation...
    # [Previous main() code with updated function calls]
    pass


if __name__ == "__main__":
    main()