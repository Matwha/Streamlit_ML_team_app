"""Model evaluation utilities."""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats


class ModelEvaluator:
    """Class for model evaluation functions."""

    @staticmethod
    def calculate_metrics(actual, predicted):
        """Calculate comprehensive set of evaluation metrics."""
        try:
            metrics = {
                'MAE': mean_absolute_error(actual, predicted),
                'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
                'MAPE': np.mean(np.abs((actual - predicted) / actual)) * 100,
                'R2': r2_score(actual, predicted)
            }

            # Additional metrics
            residuals = actual - predicted
            metrics.update({
                'Mean Error': np.mean(residuals),
                'Std Error': np.std(residuals),
                'Skewness': stats.skew(residuals),
                'Kurtosis': stats.kurtosis(residuals)
            })

            return metrics

        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")
            return None

    @staticmethod
    def test_residuals(residuals):
        """Perform statistical tests on residuals."""
        try:
            tests = {}

            # Normality test
            stat, p_value = stats.normaltest(residuals)
            tests['Normality'] = {
                'statistic': stat,
                'p_value': p_value,
                'null_hypothesis': 'Residuals are normally distributed'
            }

            # Autocorrelation test
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lbvalue, p_value = acorr_ljungbox(residuals, lags=10, return_df=False)
            tests['Autocorrelation'] = {
                'statistic': lbvalue[0],
                'p_value': p_value[0],
                'null_hypothesis': 'No autocorrelation present'
            }

            # Heteroscedasticity test
            from statsmodels.stats.diagnostic import het_breuschpagan
            bp_test = het_breuschpagan(residuals, np.ones((len(residuals), 1)))
            tests['Heteroscedasticity'] = {
                'statistic': bp_test[0],
                'p_value': bp_test[1],
                'null_hypothesis': 'Homoscedasticity present'
            }

            return tests

        except Exception as e:
            st.error(f"Error performing residual tests: {str(e)}")
            return None

    @staticmethod
    def evaluate_forecasts(forecasts, actuals):
        """Evaluate multiple forecast horizons."""
        try:
            horizon_metrics = []

            for h in range(len(forecasts)):
                metrics = {
                    'Horizon': h + 1,
                    'MAE': mean_absolute_error(actuals[h:], forecasts[h:]),
                    'RMSE': np.sqrt(mean_squared_error(actuals[h:], forecasts[h:])),
                    'MAPE': np.mean(np.abs((actuals[h:] - forecasts[h:]) / actuals[h:])) * 100
                }
                horizon_metrics.append(metrics)

            return pd.DataFrame(horizon_metrics)

        except Exception as e:
            st.error(f"Error evaluating forecasts: {str(e)}")
            return None

    @staticmethod
    def compare_models(models_results):
        """Compare multiple models statistically."""
        try:
            if len(models_results) < 2:
                return "Need at least two models to compare"

            comparisons = []
            names = list(models_results.keys())

            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    model1, model2 = names[i], names[j]

                    # Perform Diebold-Mariano test
                    from statsmodels.stats.diagnostic import compare_forecast_accuracy
                    dm_stat, dm_pvalue = compare_forecast_accuracy(
                        models_results[model1]['residuals'],
                        models_results[model2]['residuals']
                    )

                    comparisons.append({
                        'Model 1': model1,
                        'Model 2': model2,
                        'DM Statistic': dm_stat,
                        'p-value': dm_pvalue,
                        'Better Model': model1 if dm_stat < 0 else model2 if dm_stat > 0 else 'Equal'
                    })

            return pd.DataFrame(comparisons)

        except Exception as e:
            st.error(f"Error comparing models: {str(e)}")
            return None

    def display_evaluation_results(self, metrics, tests=None, horizon_metrics=None):
        """Display evaluation results in Streamlit."""
        st.subheader("Model Performance Metrics")

        # Display basic metrics
        cols = st.columns(4)
        for i, (metric, value) in enumerate(metrics.items()):
            cols[i % 4].metric(metric, f"{value:.4f}")

        # Display statistical tests if available
        if tests:
            st.subheader("Statistical Tests")
            for test_name, test_results in tests.items():
                with st.expander(f"{test_name} Test"):
                    st.write(f"Null Hypothesis: {test_results['null_hypothesis']}")
                    st.write(f"Test Statistic: {test_results['statistic']:.4f}")
                    st.write(f"P-value: {test_results['p_value']:.4f}")

    def calculate_prediction_intervals(model, data, predictions, alpha=0.05):
        """Calculate prediction intervals based on model type."""
        if hasattr(model, 'get_forecast'):  # For ARIMA/SARIMA models
            forecast = model.get_forecast(len(data))
            return forecast.conf_int(alpha=alpha)
        else:  # For ML models
            residuals = data - predictions
            std_dev = np.std(residuals)
            z_score = stats.norm.ppf(1 - alpha / 2)

            lower_bound = predictions - z_score * std_dev
            upper_bound = predictions + z_score * std_dev

            return lower_bound, upper_bound


class PredictionDisplayManager:
    """Manages prediction display settings and visualization for the Streamlit app."""

    def __init__(self):
        # Initialize session state for display settings if not exists
        if 'display_settings' not in st.session_state:
            st.session_state.display_settings = {
                'format': 'both',
                'show_ci': False,
                'ci_level': 0.95,
                'n_predictions': 10
            }

    def add_sidebar_controls(self):
        """Add display controls to Streamlit sidebar."""
        st.sidebar.header("Display Settings")

        # Display format selection
        settings = {}
        settings['display_format'] = st.sidebar.selectbox(
            "Select Display Format",
            ["Table", "Plot", "Both"],
            help="Choose how to display the predictions"
        )

        # Confidence interval controls 
        settings['show_ci'] = st.sidebar.checkbox(
            "Show Confidence Intervals",
            value=False,
            help="Display prediction confidence intervals"
        )

        if settings['show_ci']:
            settings['ci_level'] = st.sidebar.slider(
                "Confidence Level (%)",
                min_value=80,
                max_value=99,
                value=95,
                help="Set confidence interval level"
            ) / 100

        # Number of predictions to show
        settings['n_predictions'] = st.sidebar.number_input(
            "Number of Predictions to Display",
            min_value=1,
            max_value=100,
            value=10,
            help="Number of future predictions to show"
        )

        return settings

    def display_results(self, df, model_name, metrics=None):
        """Display prediction results based on current settings."""
        try:
            settings = st.session_state.get('display_settings', {})
            
            # Display format handling
            if settings.get('display_format') in ['Table', 'Both']:
                st.subheader(f"{model_name} Predictions - Table View")
                st.dataframe(df)

                # Add download button
                csv = df.to_csv(index=True)
                st.download_button(
                    label="Download Predictions",
                    data=csv,
                    file_name=f"{model_name.lower()}_predictions.csv",
                    mime='text/csv'
                )

            if settings.get('display_format') in ['Plot', 'Both']:
                st.subheader(f"{model_name} Predictions - Plot View")
                
                fig = go.Figure()

                # Add actual values if they exist
                if 'Actual' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['Actual'],
                        name='Actual',
                        line=dict(color='blue', width=2)
                    ))

                # Add predicted values
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Predicted'],
                    name='Predicted',
                    line=dict(color='red', width=2, dash='dash')
                ))

                # Add confidence intervals if requested and available
                if settings.get('show_ci', False) and 'Lower Bound' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['Upper Bound'],
                        fill=None,
                        mode='lines',
                        line_color='rgba(255,0,0,0)',
                        showlegend=False
                    ))

                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['Lower Bound'],
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(255,0,0,0)',
                        name=f"{int(settings.get('ci_level', 0.95)*100)}% Confidence Interval",
                        fillcolor='rgba(255,0,0,0.2)'
                    ))

                fig.update_layout(
                    title=f'{model_name} Time Series Forecast',
                    xaxis_title='Time',
                    yaxis_title='Value',
                    hovermode='x unified',
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)

            # Display metrics if provided
            if metrics is not None:
                self.display_metrics(metrics)

        except Exception as e:
            st.error(f"Error displaying results: {str(e)}")

    def display_metrics(self, metrics):
        """Display forecast accuracy metrics."""
        try:
            # Create columns for metrics
            cols = st.columns(len(metrics))

            # Display each metric in its own column
            for col, (metric_name, metric_value) in zip(cols, metrics.items()):
                col.metric(
                    metric_name,
                    f"{metric_value:.4f}" if metric_name != 'MAPE' else f"{metric_value:.2f}%"
                )

        except Exception as e:
            st.error(f"Error displaying metrics: {str(e)}")

    def plot_predictions(self, results_df, show_intervals=False, model_name="Model"):
        """Plot predictions with optional confidence intervals."""
        try:
            fig = go.Figure()

            # Plot actual values
            if 'Actual' in results_df.columns:
                fig.add_trace(go.Scatter(
                    x=results_df.index,
                    y=results_df['Actual'],
                    name='Actual',
                    line=dict(color='blue')
                ))

            # Plot predictions
            fig.add_trace(go.Scatter(
                x=results_df.index,
                y=results_df['Predicted'],
                name='Predicted',
                line=dict(color='red', dash='dash')
            ))

            # Add confidence intervals if available and requested
            if show_intervals and 'Lower Bound' in results_df.columns and 'Upper Bound' in results_df.columns:
                fig.add_trace(go.Scatter(
                    x=results_df.index.tolist() + results_df.index[::-1].tolist(),
                    y=results_df['Upper Bound'].tolist() + results_df['Lower Bound'][::-1].tolist(),
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval'
                ))

            fig.update_layout(
                title=f'{model_name} Predictions',
                xaxis_title='Date',
                yaxis_title='Value',
                template='plotly_white',
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error plotting predictions: {str(e)}")