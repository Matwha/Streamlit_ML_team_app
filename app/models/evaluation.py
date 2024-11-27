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