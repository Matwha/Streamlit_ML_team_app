# models/evaluation.py
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats


class ModelEvaluator:
    def __init__(self, metrics=['MAE', 'RMSE', 'MAPE']):
        self.metrics = metrics
        self.results = {}

    def evaluate_model(self, y_true, y_pred, model_name):
        metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'R2': r2_score(y_true, y_pred)
        }
        self.results[model_name] = metrics
        return metrics

    def compare_models(self):
        comparison_df = pd.DataFrame(self.results).T
        return comparison_df

    def plot_comparison(self):
        df = self.compare_models()

        fig = go.Figure()
        for metric in self.metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=df.index,
                y=df[metric],
                text=df[metric].round(3)
            ))

        fig.update_layout(
            title="Model Comparison",
            barmode='group',
            xaxis_title="Models",
            yaxis_title="Metric Value"
        )
        return fig

    # In models/evaluation.py, inside ModelEvaluator class

    def display_results(self, y_true, predictions_dict, conf_intervals=None, display_type="Both"):
        """
        Display model results based on selected display type
        """
        if display_type in ["Table", "Both"]:
            results_df = pd.DataFrame({"Actual": y_true})
            for model_name, preds in predictions_dict.items():
                results_df[f"{model_name}_Predictions"] = preds
                if conf_intervals and model_name in conf_intervals:
                    results_df[f"{model_name}_Lower"] = conf_intervals[model_name][0]
                    results_df[f"{model_name}_Upper"] = conf_intervals[model_name][1]
            st.dataframe(results_df)

        if display_type in ["Plot", "Both"]:
            fig = self.plot_predictions(y_true, predictions_dict, conf_intervals)
            st.plotly_chart(fig, use_container_width=True)

    def plot_predictions(self, actual, predictions_dict, conf_intervals=None):
        """
        Create interactive plot of model predictions
        """
        fig = go.Figure()

        # Plot actual values
        fig.add_trace(go.Scatter(
            x=actual.index,
            y=actual,
            name='Actual',
            line=dict(color=PLOT_CONFIGS['prediction']['actual_color'])
        ))

        # Plot predictions and intervals for each model
        for model_name, preds in predictions_dict.items():
            fig.add_trace(go.Scatter(
                x=preds.index,
                y=preds,
                name=f'{model_name} Predictions',
                line=dict(color=PLOT_CONFIGS['prediction']['pred_color'])
            ))

            if conf_intervals and model_name in conf_intervals:
                lower, upper = conf_intervals[model_name]
                fig.add_trace(go.Scatter(
                    x=preds.index.tolist() + preds.index[::-1].tolist(),
                    y=upper.tolist() + lower[::-1].tolist(),
                    fill='toself',
                    fillcolor=PLOT_CONFIGS['prediction']['ci_color'],
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{model_name} CI'
                ))

        fig.update_layout(
            title="Model Predictions Comparison",
            xaxis_title="Time",
            yaxis_title="Value",
            template='plotly_white'
        )

        return fig

    def plot_predictions(self, actual, predictions_dict, conf_intervals=None):
        fig = go.Figure()

        # Plot actual values
        fig.add_trace(go.Scatter(
            x=actual.index,
            y=actual,
            name='Actual',
            line=dict(color='black')
        ))

        # Plot predictions for each model
        colors = ['red', 'blue', 'green']
        for (model_name, predictions), color in zip(predictions_dict.items(), colors):
            fig.add_trace(go.Scatter(
                x=predictions.index,
                y=predictions,
                name=f'{model_name} Predictions',
                line=dict(color=color, dash='dash')
            ))

            # Add confidence intervals if available
            if conf_intervals and model_name in conf_intervals:
                lower, upper = conf_intervals[model_name]
                fig.add_trace(go.Scatter(
                    x=predictions.index.tolist() + predictions.index[::-1].tolist(),
                    y=upper.tolist() + lower[::-1].tolist(),
                    fill='toself',
                    fillcolor=f'rgba{tuple(list(colors.index(color)) + [0.2])}',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{model_name} CI'
                ))

        fig.update_layout(
            title="Model Predictions Comparison",
            xaxis_title="Date",
            yaxis_title="Value",
            template='plotly_white'
        )
        return fig


class CrossValidator:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.results = {}

    def cross_validate_model(self, model, X, y, model_name):
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model.train(X_train, y_train)
            y_pred = model.predict(X_val)

            score = mean_absolute_error(y_val, y_pred)
            scores.append(score)

        self.results[model_name] = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'scores': scores
        }
        return self.results[model_name]

    def plot_cv_results(self):
        fig = go.Figure()

        for model_name, result in self.results.items():
            fig.add_trace(go.Box(
                y=result['scores'],
                name=model_name,
                boxpoints='all'
            ))

        fig.update_layout(
            title="Cross-Validation Results",
            yaxis_title="MAE Score",
            template='plotly_white'
        )
        return fig


def compare_model_residuals(models_dict, actual):
    fig = make_subplots(rows=2, cols=len(models_dict),
                        subplot_titles=[f"{name} Residuals" for name in models_dict.keys()])

    for i, (name, model) in enumerate(models_dict.items(), 1):
        residuals = actual - model.predict(actual.index)

        # Residual plot
        fig.add_trace(
            go.Scatter(y=residuals, mode='markers', name=f'{name} Residuals'),
            row=1, col=i
        )

        # QQ plot
        qq = stats.probplot(residuals)
        fig.add_trace(
            go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers',
                       name=f'{name} Q-Q Plot'),
            row=2, col=i
        )

    fig.update_layout(height=800, title_text="Model Residuals Analysis")
    return fig

def plot_predictions(actual, predictions, intervals=None, model_name="Model"):
    """
    Create an interactive plot of predictions with optional confidence intervals
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predictions : array-like
        Predicted values
    intervals : tuple, optional
        Tuple of (lower_bound, upper_bound) for confidence intervals
    model_name : str
        Name of the model for plot title
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Interactive plot
    """
    fig = go.Figure()

    # Plot actual values
    fig.add_trace(go.Scatter(
        name='Actual',
        y=actual,
        mode='lines',
        line=dict(color='blue')
    ))

    # Plot predictions
    fig.add_trace(go.Scatter(
        name='Predicted',
        y=predictions,
        mode='lines',
        line=dict(color='red', dash='dash')
    ))

    # Add confidence intervals if provided
    if intervals is not None:
        lower_bound, upper_bound = intervals
        fig.add_trace(go.Scatter(
            name='Upper Bound',
            y=upper_bound,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            name='Lower Bound',
            y=lower_bound,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        ))

    fig.update_layout(
        title=f'{model_name} Predictions',
        yaxis_title='Value',
        xaxis_title='Time',
        hovermode='x'
    )

    return fig

def plot_model_comparison(selected_models, selected_metrics, session_state):
    """
    Create a comparison plot for multiple models using selected metrics.
    
    Parameters:
    -----------
    selected_models : list
        List of model names to compare
    selected_metrics : list 
        List of metrics to show in comparison
    session_state : SessionState
        Streamlit session state containing model results
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Interactive comparison plot
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create figure with secondary y-axis
        fig = make_subplots(rows=len(selected_metrics), cols=1,
                          subplot_titles=selected_metrics)
        
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        
        for i, metric in enumerate(selected_metrics, start=1):
            for j, (model_name, color) in enumerate(zip(selected_models, colors)):
                if f"{model_name}_results" in session_state:
                    results = session_state[f"{model_name}_results"]
                    if metric in results:
                        fig.add_trace(
                            go.Scatter(
                                y=[results[metric]],
                                name=f"{model_name} - {metric}",
                                marker=dict(color=color),
                                showlegend=True if i == 1 else False
                            ),
                            row=i, col=1
                        )

        # Update layout
        fig.update_layout(
            height=300 * len(selected_metrics),
            title_text="Model Performance Comparison",
            template='plotly_white'
        )

        # Update y-axes labels
        for i, metric in enumerate(selected_metrics, start=1):
            fig.update_yaxes(title_text=metric, row=i, col=1)

        return fig

    except Exception as e:
        st.error(f"Error in plot_model_comparison: {str(e)}")
        return None