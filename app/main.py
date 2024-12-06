"""Main Streamlit application for time series forecasting."""
import streamlit as st
import plotly.graph_objects as go
import tensorflow as tf
import pandas as pd
import numpy as np
from models.trainer import ModelTrainer
from dataclasses import dataclass
from typing import Optional, Dict, Any
from data.preprocessor import TimeSeriesPreprocessor

from config import APP_TITLE, APP_DESCRIPTION, DEFAULT_GITHUB_REPO
from data.loader import DataLoader
from data.data_analyzer import TimeSeriesAnalyzer
from models.evaluation import PredictionDisplayManager
from models.traditional import create_traditional_model
from models.machine_learning import create_ml_model
from models.deep_learning import SimpleRNNModel, LSTMModel, StackedModel, SequentialModel
from utils.helpers import (
    setup_environment,
    validate_data,
    setup_model_selection,
    get_model_parameters,
    display_data_analysis_tabs
)


@dataclass
class AppState:
    """Class to store application state."""
    analysis_complete: bool = False
    selected_model: Optional[str] = None
    target_column: Optional[str] = None
    trained_model: Any = None
    results: Optional[Dict] = None


class TimeSeriesApp:
    """Main application class for time series forecasting."""

    def __init__(self):
        """Initialize application components."""
        setup_environment()
        self.data_loader = DataLoader()
        self.display_manager = PredictionDisplayManager()

        # Initialize session state
        if 'app_state' not in st.session_state:
            st.session_state.app_state = AppState()

    def setup_page(self):
        """Configure page settings and display title."""
        st.set_page_config(layout="wide", page_title=APP_TITLE)
        st.title(APP_TITLE)
        st.markdown(APP_DESCRIPTION)

    def load_data(self) -> Optional[pd.DataFrame]:
        """Handle data loading from various sources."""
        st.sidebar.header("1. Data Source")
        data_source = st.sidebar.radio(
            "Select data source",
            ["GitHub Repository", "Upload File", "Example Data"]
        )

        df = None
        if data_source == "GitHub Repository":
            github_url = st.sidebar.text_input("GitHub repository URL", DEFAULT_GITHUB_REPO)
            if github_url:
                csv_files = self.data_loader.get_github_files(github_url)
                if csv_files:
                    selected_file = st.sidebar.selectbox(
                        "Select CSV file",
                        [file[0] for file in csv_files]
                    )
                    if selected_file:
                        file_url = next(file[1] for file in csv_files
                                        if file[0] == selected_file)
                        df = self.data_loader.load_data(
                            "github",
                            github_url=github_url,
                            selected_file=file_url
                        )

        elif data_source == "Upload File":
            uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_file:
                df = self.data_loader.load_data("upload", uploaded_file=uploaded_file)

        else:  # Example Data
            df = self.data_loader.load_example_data()
            st.sidebar.info("Using example data")

        return df

    def select_target_column(self, df: pd.DataFrame) -> Optional[str]:
        """Allow user to select target column for analysis."""
        target_column = st.sidebar.selectbox(
            "Select Target Column",
            df.columns.tolist()
        )
        return target_column

    def analyze_data(self, df: pd.DataFrame, target_column: str):
        """Perform data analysis and display results."""
        st.header("2. Data Analysis")
        analyzer = TimeSeriesAnalyzer(df)
        display_data_analysis_tabs(analyzer, target_column)

    def setup_model(self):
        """Configure model selection and parameters."""
        st.header("3. Model Selection")
        available_models, model_name = setup_model_selection()

        if model_name is None:
            return None, None

        # Get model parameters based on selection
        params = get_model_parameters(model_name)

        # Ensure params has the correct structure
        if params is not None and not isinstance(params, dict):
            params = {'build_params': params, 'train_params': {}}
        elif params is not None and 'build_params' not in params:
            params = {'build_params': params, 'train_params': {}}

        return model_name, params

    def create_model(self, model_name: str, model_params: Dict):
        """Create model instance based on model type."""
        try:
            if model_name in ['Simple RNN', 'LSTM', 'Stacked LSTM+RNN', 'Sequential']:
                # Extract initialization parameters
                init_params = {
                    'sequence_length': model_params['build_params'].pop('sequence_length', 10),
                    'n_features': model_params['build_params'].pop('n_features', 1)
                }

                model_class = {
                    'Simple RNN': SimpleRNNModel,
                    'LSTM': LSTMModel,
                    'Stacked LSTM+RNN': StackedModel,
                    'Sequential': SequentialModel
                }[model_name]

                # Create model instance with init params
                model = model_class(name=model_name, **init_params)

                # Build model with build parameters only
                model.build(**model_params['build_params'])

                # Store training parameters in the model instance for later use
                model.train_params = model_params['train_params']

                return model

            elif model_name in ['ARIMA', 'SARIMA']:
                return create_traditional_model(model_name, **model_params['build_params'])
            else:
                return create_ml_model(model_name, **model_params['build_params'])

        except Exception as e:
            st.error(f"Error creating model: {str(e)}")
            return None

    # In main.py, replace or modify the train_and_evaluate method:

    def train_and_evaluate(self, df: pd.DataFrame, target_column: str, model_name: str, model_params: Dict):
        """Train model and evaluate results."""
        try:
            # Create model instance
            model = self.create_model(model_name, model_params)
            if model is None:
                return None, None, None

            # Create trainer instance
            trainer = ModelTrainer(model, target_column)

            # Train and evaluate
            return trainer.train_and_evaluate(df, model_name, model_params)

        except Exception as e:
            st.error(f"Error in model training and evaluation: {str(e)}")
            with st.expander("🔍 Debug: Error Details", expanded=True):
                st.error(f"Error Type: {type(e).__name__}")
                st.error(f"Error Message: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
            return None, None, None

    def run(self):
        """Run the main application."""
        self.setup_page()

        # Load data
        df = self.load_data()
        if df is None or not validate_data(df)[0]:
            return

        # Select target column
        target_column = self.select_target_column(df)
        if target_column is None:
            return

        # Initialize preprocessor and get options
        preprocessor = TimeSeriesPreprocessor()
        preprocessing_options = preprocessor.add_preprocessing_controls()

        # Analyze original data before preprocessing
        self.analyze_data(df, target_column)

        # Setup model
        model_name, model_params = self.setup_model()
        if model_name is None:
            return

        # Add display and prediction controls
        display_options = st.sidebar.expander("Display & Prediction Options", expanded=True)
        with display_options:
            # Display settings
            display_type = st.selectbox(
                "Display Predictions As",
                ["Table", "Graph", "Both"]
            )

            # Confidence interval settings
            show_intervals = st.checkbox("Show Confidence Intervals", value=False)
            if show_intervals:
                confidence_level = st.slider(
                    "Confidence Level",
                    min_value=0.8,
                    max_value=0.99,
                    value=0.95,
                    step=0.01
                )
                interval_method = st.selectbox(
                    "Interval Method",
                    ["Bootstrap", "Quantile", "Analytical"]
                )
                if interval_method == "Bootstrap":
                    n_iterations = st.slider(
                        "Bootstrap Iterations",
                        min_value=100,
                        max_value=1000,
                        value=500,
                        step=100
                    )

                # Add interval settings to model parameters
                if 'train_params' not in model_params:
                    model_params['train_params'] = {}

                model_params['train_params'].update({
                    'return_intervals': show_intervals,
                    'confidence_level': confidence_level if show_intervals else None,
                    'interval_method': interval_method if show_intervals else None,
                    'n_iterations': n_iterations if show_intervals and interval_method == "Bootstrap" else None
                })

        # Additional display controls from display manager
        display_settings = self.display_manager.add_sidebar_controls()

        # Store display preferences in session state
        if 'display_settings' not in st.session_state:
            st.session_state.display_settings = {}

        st.session_state.display_settings.update({
            'display_type': display_type,
            'show_intervals': show_intervals,
            'confidence_level': confidence_level if show_intervals else None,
            'interval_method': interval_method if show_intervals else None,
            'n_iterations': n_iterations if show_intervals and interval_method == "Bootstrap" else None
        })

        # Train and evaluate
        if st.button("Run Analysis"):
            with st.spinner("Running analysis..."):
                # Train and evaluate with original dataframe
                model, results_df, metrics = self.train_and_evaluate(
                    df=df,
                    target_column=target_column,
                    model_name=model_name,
                    model_params=model_params
                )

                if model is not None:
                    st.session_state.app_state.analysis_complete = True
                    st.session_state.app_state.trained_model = model
                    st.session_state.app_state.results = {
                        'df': results_df,
                        'metrics': metrics
                    }

                    # Display results based on selected display type
                    if display_type in ["Table", "Both"]:
                        st.subheader("Prediction Results - Table View")
                        st.dataframe(results_df)

                    if display_type in ["Graph", "Both"]:
                        st.subheader("Prediction Results - Graph View")
                        self.display_manager.plot_predictions(
                            results_df,
                            show_intervals=show_intervals,
                            model_name=model_name
                        )

                    # Display metrics
                    st.subheader("Model Performance Metrics")
                    self.display_manager.display_metrics(metrics)

                    st.success("Analysis completed successfully!")


if __name__ == "__main__":
    app = TimeSeriesApp()
    app.run()