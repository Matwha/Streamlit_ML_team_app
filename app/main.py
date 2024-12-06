"""Main Streamlit application for time series forecasting."""
import tensorflow as tf
from dataclasses import dataclass
from typing import Optional, Dict, Any
from tensorflow import keras
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from models.traditional import ARIMAModel, SARIMAModel
from models.machine_learning import RandomForestModel, XGBoostModel
from models.deep_learning import SimpleRNNModel, LSTMModel, StackedModel
from config import (TRADITIONAL_MODELS, ML_MODELS, DL_MODELS, APP_TITLE, 
                   APP_DESCRIPTION, DEFAULT_GITHUB_REPO, MODEL_CATEGORIES,
                   EXAMPLE_DATA_URL)  # Add this import
import streamlit as st
import pandas as pd
from data.loader import DataLoader
from utils.visualization import DataVisualizer
from utils.helpers import setup_environment, validate_data
from data.data_analyzer import TimeSeriesAnalyzer


setup_environment()

# Replace the existing AnalysisState class and main() definition with:

@dataclass
class AnalysisState:
    """Class to store analysis state"""
    results: Optional[Dict[str, Any]] = None
    display_format: str = "Table"
    show_ci: bool = False
    ci_level: float = 95.0
    has_run: bool = False

def main():
    """Main application function"""
    st.set_page_config(layout="wide", page_title="Time Series Forecasting App")
    st.title(APP_TITLE)
    st.markdown(APP_DESCRIPTION)

    # Initialize classes
    data_loader = DataLoader()
    data_visualizer = DataVisualizer()

    # Initialize session state if needed
    if 'analysis_state' not in st.session_state:
        st.session_state.analysis_state = AnalysisState()

    # Data Source Selection
    st.sidebar.header("1. Data Source")
    data_source = st.sidebar.radio(
        "Select data source",
        ["GitHub Repository", "Upload File", "Example Data"]
    )

    df = None
    if data_source == "GitHub Repository":
        github_url = st.sidebar.text_input("GitHub repository URL", DEFAULT_GITHUB_REPO)
        if github_url:
            csv_files = data_loader.get_github_files(github_url)
            if csv_files:
                selected_file = st.sidebar.selectbox("Select CSV file",
                                                   [file[0] for file in csv_files])
                if selected_file:
                    file_url = next(file[1] for file in csv_files if file[0] == selected_file)
                    df = data_loader.load_data("github", github_url=github_url, selected_file=file_url)

    elif data_source == "Upload File":
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file:
            df = data_loader.load_data("upload", uploaded_file=uploaded_file)

    else:  # Example Data
        df = data_loader.load_example_data(EXAMPLE_DATA_URL)
        st.sidebar.info("Using example stock price data")

    if df is not None and data_loader.validate_data(df):
        # Add target column selection
        target_column = st.sidebar.selectbox(
            "Select Target Column",
            df.columns.tolist()
        )

        # Data Analysis Section
        st.header("2. Data Analysis")

        # Initialize TimeSeriesAnalyzer
        analyzer = TimeSeriesAnalyzer(df)

        analysis_tabs = st.tabs([
            "Basic Statistics",
            "Time Series Properties",
            "Data Quality",
            "Feature Engineering"
        ])

        with analysis_tabs[0]:
            st.subheader("Basic Statistical Analysis")
            stats = analyzer.analyze_basic_stats(target_column)

            # Display basic info first
            st.write("##### Data Information")
            st.write(f"- Data Type: {stats['Data Type']}")
            st.write(f"- Is Numeric: {stats['Is Numeric']}")

            if not stats['Is Numeric']:
                if 'Start Date' in stats:  # Check if it's a datetime column
                    st.write("##### Date Range Information")
                    st.write(f"- Start Date: {stats['Start Date']}")
                    st.write(f"- End Date: {stats['End Date']}")
                    st.write(f"- Date Range (days): {stats['Date Range (days)']}")
                    st.write(f"- Number of Unique Dates: {stats['Number of Unique Dates']}")
                    if 'Missing Values' in stats:
                        st.write(f"- Missing Values: {stats['Missing Values']}")
                    if 'Note' in stats:
                        st.info(stats['Note'])

                    # Add visualization for date-based data
                    st.write("##### Data Analysis Visualization")
                    try:
                        analyzer.plot_analysis_results(target_column)
                    except Exception as e:
                        st.warning("Could not generate visualizations. Some analyses may not be applicable to date columns.")
                else:
                    st.warning("Selected column contains non-numeric data. Limited analysis available.")
                    st.write("##### Sample Values")
                    st.write(stats['Sample Values'])
                    if 'Most Common Values' in stats:
                        st.write("##### Most Common Values")
                        st.write(stats['Most Common Values'])
                if 'Note' in stats:
                    st.info(stats['Note'])
            else:
                # Display numeric statistics
                numeric_stats = {k: v for k, v in stats.items()
                               if k not in ['Data Type', 'Is Numeric', 'Sample Values', 'Note']}
                st.write("##### Statistical Measures")
                st.write(pd.DataFrame([numeric_stats]).T)

                # Add visualization for numeric data
                st.write("##### Data Analysis Visualization")
                analyzer.plot_analysis_results(target_column)

        with analysis_tabs[1]:
            st.subheader("Time Series Properties")

            if not stats['Is Numeric']:
                st.warning("Time series analysis requires numeric data. Please select a numeric column.")
            else:
                # Stationarity Analysis
                st.write("##### Stationarity Tests")
                stationarity = analyzer.check_stationarity(target_column)

                col1, col2 = st.columns(2)
                with col1:
                    st.write("ADF Test Results:")
                    st.write(f"- Statistic: {stationarity['ADF']['statistic']:.4f}")
                    st.write(f"- p-value: {stationarity['ADF']['p-value']:.4f}")
                    st.write(f"- Is Stationary: {stationarity['ADF']['is_stationary']}")

                with col2:
                    st.write("KPSS Test Results:")
                    st.write(f"- Statistic: {stationarity['KPSS']['statistic']:.4f}")
                    st.write(f"- p-value: {stationarity['KPSS']['p-value']:.4f}")
                    st.write(f"- Is Stationary: {stationarity['KPSS']['is_stationary']}")

                if stationarity['recommendations']:
                    st.write("##### Recommendations:")
                    for rec in stationarity['recommendations']:
                        st.write(f"- {rec}")

                # Seasonality Analysis
                st.write("##### Seasonality Analysis")
                seasonality = analyzer.detect_seasonality(target_column)
                st.write(f"- Has Seasonality: {seasonality['has_seasonality']}")
                st.write(f"- Seasonal Strength: {seasonality['seasonal_strength']:.4f}")
                st.write(f"- Suggested Period: {seasonality['suggested_period']}")

        with analysis_tabs[2]:
            st.subheader("Data Quality Analysis")

            # Missing Values Analysis
            st.write("##### Missing Values Analysis")
            missing = analyzer.analyze_missing_values(target_column)
            st.write(f"- Total Missing: {missing['total_missing']}")
            st.write(f"- Percentage Missing: {missing['percentage_missing']:.2f}%")

            if missing['recommendations']:
                st.write("Imputation Recommendations:")
                for rec in missing['recommendations']:
                    st.write(f"- {rec}")

            # Outlier Analysis
            if stats['Is Numeric']:
                st.write("##### Outlier Analysis")
                outliers = analyzer.detect_outliers(target_column)
                st.write(f"- Number of Outliers: {outliers['count']}")
                st.write(f"- Percentage of Outliers: {outliers['percentage']:.2f}%")
                st.write(f"- Detection Method: {outliers['method']}")
            else:
                st.info("Outlier analysis is only available for numeric columns.")

        with analysis_tabs[3]:
            st.subheader("Feature Engineering Recommendations")

            if not stats['Is Numeric']:
                st.warning("Feature engineering recommendations require numeric data.")
            else:
                recommendations = analyzer.generate_feature_recommendations(target_column)

                for rec in recommendations:
                    st.write(f"##### {rec['type'].replace('_', ' ').title()}")
                    st.write(rec['description'])

                    if 'suggested_windows' in rec:
                        st.write("Suggested window sizes:", rec['suggested_windows'])
                    if 'suggested_lags' in rec:
                        st.write("Suggested lag values:", rec['suggested_lags'])
                    if 'suggested_differences' in rec:
                        st.write("Suggested differences:", rec['suggested_differences'])

        # Visualization
        st.header("Data Analysis Visualization")
        if stats['Is Numeric']:
            analyzer.plot_analysis_results(target_column)
        else:
            st.warning("Visualizations are only available for numeric columns.")

        # Model Analysis Section
        st.header("3. Model Selection")

        # Only show model selection if we have numeric data or datetime data
        if not stats['Is Numeric'] and 'Start Date' not in stats:
            st.warning("Model analysis requires numeric or datetime data. Please select an appropriate column.")
            return

        # Model category selection
        selected_categories = st.sidebar.multiselect(
            "Select Model Categories",
            MODEL_CATEGORIES,
            default=['Traditional']
        )

        # Build available models list based on selected categories
        available_models = []
        if 'Traditional' in selected_categories:
            available_models.extend(TRADITIONAL_MODELS)
        if 'Machine Learning' in selected_categories:
            available_models.extend(ML_MODELS)
        if 'Deep Learning' in selected_categories:
            available_models.extend(DL_MODELS)

        # Initialize model_name
        model_name = None

        if not available_models:
            st.warning("Please select at least one model category.")
        else:
            # Model selection dropdown
            model_name = st.sidebar.selectbox(
                "Select Model",
                available_models
            )

            # Model Parameters section
            model_params = {}
            with st.sidebar.expander(f"{model_name} Parameters", expanded=True):
                if model_name == 'ARIMA':
                    st.subheader("ARIMA Parameters")
                    model_params[model_name] = {
                        'p': st.slider(f'AR order (p)', 0, 5, 1),
                        'd': st.slider(f'Differencing order (d)', 0, 2, 1),
                        'q': st.slider(f'MA order (q)', 0, 5, 1)
                    }
                elif model_name == 'SARIMA':
                    st.subheader("SARIMA Parameters")
                    model_params[model_name] = {
                        'p': st.slider(f'AR order (p)', 0, 5, 1),
                        'd': st.slider(f'Differencing order (d)', 0, 2, 1),
                        'q': st.slider(f'MA order (q)', 0, 5, 1),
                        'P': st.slider(f'Seasonal AR order (P)', 0, 2, 1),
                        'D': st.slider(f'Seasonal differencing (D)', 0, 1, 1),
                        'Q': st.slider(f'Seasonal MA order (Q)', 0, 2, 1),
                        's': st.select_slider(f'Seasonal period (s)', options=[4, 7, 12], value=12)
                    }
                elif model_name == 'Random Forest':
                    st.subheader("Random Forest Parameters")
                    model_params[model_name] = {
                        'n_estimators': st.slider('Number of Trees', 10, 200, 100),
                        'max_depth': st.slider('Max Tree Depth', 2, 30, 10),
                        'min_samples_split': st.slider('Min Samples Split', 2, 20, 2),
                        'random_state': 42
                    }
                elif model_name == 'XGBoost':
                    st.subheader("XGBoost Parameters")
                    model_params[model_name] = {
                        'n_estimators': st.slider('Number of Trees', 10, 200, 100),
                        'max_depth': st.slider('Max Tree Depth', 2, 30, 6),
                        'learning_rate': st.slider('Learning Rate', 0.01, 0.3, 0.1, 0.01),
                        'random_state': 42
                    }
                elif model_name in ['Simple RNN', 'LSTM', 'Stacked LSTM+RNN']:
                    st.subheader(f"{model_name} Parameters")
                    model_params[model_name] = {
                        'sequence_length': st.slider('Sequence Length', 5, 50, 10),
                        'units': st.slider('Hidden Units', 8, 128, 32, 8),
                        'dropout_rate': st.slider('Dropout Rate', 0.0, 0.5, 0.1, 0.1),
                        'epochs': st.slider('Training Epochs', 10, 200, 50),
                        'batch_size': st.select_slider('Batch Size', options=[16, 32, 64, 128], value=32)
                    }
                elif model_name == 'Sequential':
                    st.subheader("Sequential Model Parameters")

                    # Base parameters
                    sequence_length = st.slider('Sequence Length', 5, 50, 10)
                    learning_rate = st.slider('Learning Rate', 0.0001, 0.01, 0.001, 0.0001)

                    # Layer configuration
                    n_layers = st.slider('Number of Layers', 1, 5, 2)
                    layer_configs = []

                    for i in range(n_layers):
                        st.write(f"Layer {i + 1}")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            layer_type = st.selectbox(
                                f'Type {i + 1}',
                                ['Dense', 'LSTM', 'GRU'],
                                key=f'layer_type_{i}'
                            )
                        with col2:
                            units = st.number_input(
                                f'Units {i + 1}',
                                min_value=1,
                                max_value=256,
                                value=64,
                                key=f'units_{i}'
                            )
                        with col3:
                            activation = st.selectbox(
                                f'Activation {i + 1}',
                                ['relu', 'tanh', 'sigmoid'],
                                key=f'activation_{i}'
                            )
                        with col4:
                            dropout = st.slider(
                                f'Dropout {i + 1}',
                                0.0, 0.5, 0.1,
                                key=f'dropout_{i}'
                            )
                        layer_configs.append({
                            'type': layer_type,
                            'units': units,
                            'activation': activation,
                            'dropout': dropout
                        })

                    model_params[model_name] = {
                        'sequence_length': sequence_length,
                        'learning_rate': learning_rate,
                        'layer_configs': layer_configs,
                        'epochs': st.slider('Training Epochs', 10, 200, 50),
                        'batch_size': st.select_slider('Batch Size', options=[16, 32, 64, 128], value=32)
                    }

# Initialize analysis state
            if not hasattr(st.session_state, 'analysis_state'):
                st.session_state.analysis_state = AnalysisState()
                st.session_state.analysis_state.display_format = "Table"
                st.session_state.analysis_state.show_ci = False
                st.session_state.analysis_state.ci_level = 95.0
                st.session_state.analysis_state.has_run = False
                st.session_state.analysis_state.results = None

            # Display Settings
            st.sidebar.header("4. Display Settings")

            # Add display format selection (persisting previous selection)
            display_format = st.sidebar.selectbox(
                "Select Display Format",
                ["Table", "Plot", "Both"],
                index=["Table", "Plot", "Both"].index(st.session_state.analysis_state.display_format)
            )
            st.session_state.analysis_state.display_format = display_format

            # Add confidence interval option (persisting previous selection)
            show_ci = st.sidebar.checkbox(
                "Show Confidence Intervals",
                value=st.session_state.analysis_state.show_ci
            )
            st.session_state.analysis_state.show_ci = show_ci

            if show_ci:
                ci_level = st.sidebar.slider(
                    "Confidence Level (%)",
                    80, 99,
                    value=int(st.session_state.analysis_state.ci_level),
                    key='ci_slider'
                )
                st.session_state.analysis_state.ci_level = float(ci_level)

            # Add number of future predictions
            n_predictions = st.sidebar.number_input(
                "Number of Future Predictions",
                min_value=1,
                max_value=100,
                value=10
            )

            # Run Analysis Button
            run_analysis = st.button("Run Analysis")
            if run_analysis:
                st.session_state.analysis_state.has_run = True

            if st.session_state.analysis_state.has_run:
                try:
                    st.info(f"Starting {model_name} analysis...")

                    # Prepare data based on model type
                    train_data = None
                    test_data = None
                    if model_name in TRADITIONAL_MODELS:
                        train_data, test_data = data_loader.prepare_traditional_data(df, target_column)
                    elif model_name in ML_MODELS:
                        train_data, test_data = data_loader.prepare_ml_data(df, target_column)
                    elif model_name in DL_MODELS:
                        train_data, test_data = data_loader.prepare_dl_data(df, target_column)

                    if train_data is not None and test_data is not None:
                        # Initialize model
                        model = None
                        if model_name == 'ARIMA':
                            model = ARIMAModel()
                            model.build(**model_params[model_name])
                        elif model_name == 'SARIMA':
                            model = SARIMAModel()
                            model.build(**model_params[model_name])

                        if model is not None:
                            # Train model
                            st.write("Training model...")
                            model = model.train(train_data)

                            if model is not None:
                                # Make predictions
                                st.write("Making predictions...")
                                predictions = model.predict(len(test_data))

                                if predictions is not None:
                                    # Create results DataFrame
                                    results_df = pd.DataFrame({
                                        'Actual': test_data,
                                        'Predicted': predictions
                                    })

                                    # Calculate confidence intervals if requested
                                    if show_ci:
                                        lower, upper = model.calculate_prediction_intervals(ci_level/100)
                                        if lower is not None and upper is not None:
                                            results_df['Lower Bound'] = lower
                                            results_df['Upper Bound'] = upper

                                    # Store results in session state
                                    st.session_state.analysis_state.results = {
                                        'df': results_df,
                                        'metrics': model.evaluate(test_data)
                                    }

                                    # Display results based on format selection
                                    if display_format in ["Table", "Both"]:
                                        st.write("Predictions:")
                                        st.dataframe(results_df)

                                    if display_format in ["Plot", "Both"]:
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        ax.plot(test_data.index, test_data, label='Actual', color='blue')
                                        ax.plot(test_data.index, predictions, label='Predicted', color='red')

                                        if show_ci and 'Lower Bound' in results_df.columns:
                                            ax.fill_between(test_data.index,
                                                          results_df['Lower Bound'],
                                                          results_df['Upper Bound'],
                                                          color='red', alpha=0.1,
                                                          label=f'{ci_level}% Confidence Interval')

                                        ax.set_title(f'{model_name} Predictions')
                                        ax.legend()
                                        st.pyplot(fig)

                                    # Display metrics
                                    if st.session_state.analysis_state.results['metrics']:
                                        st.write("\nModel Performance:")
                                        metrics = st.session_state.analysis_state.results['metrics']
                                        for metric, value in metrics.items():
                                            st.metric(metric, f"{value:.4f}")
                                else:
                                    st.error("Error generating predictions")
                            else:
                                st.error("Error training model")
                        else:
                            st.error(f"Model {model_name} not implemented yet")
                    else:
                        st.error("Error preparing data for analysis")

                    st.success("Analysis completed!")

                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")

if __name__ == "__main__":
    main()