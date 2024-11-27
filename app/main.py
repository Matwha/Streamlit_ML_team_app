"""Main Streamlit application for time series forecasting."""
# In main.py
import plotly.express as px
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
import os
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from config import *
from data.loader import get_github_files, load_data, load_example_data
from data.preprocessor import TimeSeriesPreprocessor
from models.deep_learning import SimpleRNNModel, LSTMModel, StackedModel
from models.trainer import ModelTrainer
from utils.visualization import DataVisualizer
from utils.helpers import validate_data, get_time_index, setup_logging, create_download_button
from utils.helpers import setup_environment
from data.loader import load_data, get_dtypes_info
setup_environment()

# Environment settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    """Main application function."""
    setup_logging()
    st.set_page_config(layout="wide", page_title="Time Series Forecasting App")

    # Title and description
    st.title(APP_TITLE)
    st.markdown(APP_DESCRIPTION)

    # Sidebar - Data Source Selection
    st.sidebar.header("Data Source")

    data_source = st.sidebar.radio(
        "Select data source",
        ["GitHub Repository", "Upload File", "Example Data"]
    )

    # Initialize df as None
    df = None

    if data_source == "GitHub Repository":
        github_url = st.sidebar.text_input(
            "GitHub repository URL",
            "https://github.com/PJalgotrader/Deep_forecasting-USU/tree/main/data"
        )
        if github_url:
            csv_files = get_github_files(github_url)
            if csv_files:
                selected_file = st.sidebar.selectbox(
                    "Select a CSV file",
                    [file[0] for file in csv_files]
                )
                if selected_file:
                    file_url = next(file[1] for file in csv_files if file[0] == selected_file)
                    df = load_data("github", github_url=github_url, selected_file=file_url)

    elif data_source == "Upload File":
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file:
            df = load_data("upload", uploaded_file=uploaded_file)

    # If data is loaded successfully, display info and continue
    # After data is loaded successfully but before model selection
    if df is not None:
        st.header("Data Exploration & Preprocessing")

        # Data Overview
        with st.expander("📊 Data Overview", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", df.shape[0])
            with col2:
                st.metric("Features", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())

            st.write("Data Sample:")
            st.dataframe(df.head())

            st.write("Data Types:")
            st.dataframe(pd.DataFrame(df.dtypes, columns=['Data Type']))

        # Time Index Settings
        with st.expander("⏰ Time Index Settings"):
            time_col = st.selectbox(
                "Select Time Index Column",
                df.columns,
                index=0 if 'date' in df.columns[0].lower() else 0
            )
            if st.button("Set Time Index"):
                try:
                    df.index = pd.to_datetime(df[time_col])
                    df = df.drop(columns=[time_col])
                    st.success(f"Set {time_col} as time index")
                except Exception as e:
                    st.error(f"Error setting time index: {e}")

        # Target Variable Selection
        with st.expander("🎯 Target Variable Selection"):
            target_col = st.selectbox(
                "Select Target Variable",
                [col for col in df.columns if col != time_col],
                index=0
            )

            st.write("Target Variable Statistics:")
            st.dataframe(df[target_col].describe())

            # Plot target variable
            fig = px.line(df, y=target_col, title=f'{target_col} Time Series')
            st.plotly_chart(fig)

        # Data Preprocessing Options
        with st.expander("🔧 Data Preprocessing"):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Handle Missing Values")
                missing_method = st.selectbox(
                    "Missing Values Strategy",
                    ["Drop", "Forward Fill", "Backward Fill", "Linear Interpolation", "Mean"]
                )

                st.subheader("Handle Outliers")
                outlier_method = st.selectbox(
                    "Outlier Strategy",
                    ["None", "IQR Method", "Z-Score Method", "Winsorization"]
                )

            with col2:
                st.subheader("Data Transformation")
                transform_method = st.selectbox(
                    "Transformation Method",
                    ["None", "Log", "Square Root", "Box-Cox", "Standardization", "Min-Max Scaling"]
                )

                st.subheader("Decomposition")
                decomposition = st.selectbox(
                    "Decomposition Method",
                    ["None", "Seasonal", "Trend", "Residual"]
                )

        # Apply Preprocessing
        if st.button("Apply Preprocessing"):
            try:
                # Handle missing values
                if missing_method == "Drop":
                    df = df.dropna()
                elif missing_method == "Forward Fill":
                    df = df.fillna(method='ffill')
                elif missing_method == "Backward Fill":
                    df = df.fillna(method='bfill')
                elif missing_method == "Linear Interpolation":
                    df = df.interpolate(method='linear')
                elif missing_method == "Mean":
                    df = df.fillna(df.mean())

                # Handle outliers
                if outlier_method == "IQR Method":
                    Q1 = df[target_col].quantile(0.25)
                    Q3 = df[target_col].quantile(0.75)
                    IQR = Q3 - Q1
                    df = df[(df[target_col] >= Q1 - 1.5 * IQR) & (df[target_col] <= Q3 + 1.5 * IQR)]
                elif outlier_method == "Z-Score Method":
                    z_scores = abs(stats.zscore(df[target_col]))
                    df = df[z_scores < 3]
                elif outlier_method == "Winsorization":
                    df[target_col] = stats.mstats.winsorize(df[target_col], limits=[0.05, 0.05])

                # Apply transformations
                if transform_method == "Log":
                    df[target_col] = np.log1p(df[target_col])
                elif transform_method == "Square Root":
                    df[target_col] = np.sqrt(df[target_col])
                elif transform_method == "Box-Cox":
                    df[target_col], _ = stats.boxcox(df[target_col])
                elif transform_method == "Standardization":
                    df[target_col] = (df[target_col] - df[target_col].mean()) / df[target_col].std()
                elif transform_method == "Min-Max Scaling":
                    df[target_col] = (df[target_col] - df[target_col].min()) / (
                                df[target_col].max() - df[target_col].min())

                # Apply decomposition
                if decomposition != "None":
                    decomposition_result = seasonal_decompose(df[target_col], period=12)
                    if decomposition == "Seasonal":
                        df[target_col] = decomposition_result.seasonal
                    elif decomposition == "Trend":
                        df[target_col] = decomposition_result.trend
                    elif decomposition == "Residual":
                        df[target_col] = decomposition_result.resid

                st.success("Preprocessing completed successfully!")

                # Show preprocessed data
                st.write("Preprocessed Data Preview:")
                st.dataframe(df.head())

                # Plot preprocessed data
                fig = px.line(df, y=target_col, title=f'Preprocessed {target_col} Time Series')
                st.plotly_chart(fig)

            except Exception as e:
                st.error(f"Error during preprocessing: {e}")

        # Display data types info
        st.subheader("Data Types Information")
        st.dataframe(get_dtypes_info(df))

    else:  # Example Data
        df = load_example_data(EXAMPLE_DATA_URL)
        st.sidebar.info("Using example stock price data")

    # Proceed if data is valid
    if validate_data(df, MIN_DATA_POINTS):
        # Data visualization
        visualizer = DataVisualizer()
        visualizer.display_data_info(df)

        if isinstance(df, pd.DataFrame):
            target_column = st.sidebar.selectbox(
                "Select target variable",
                df.columns
            )
        else:
            target_column = 'GLD'

        # Model Selection and Parameters
        st.sidebar.header("Model Selection")
        model_categories = st.sidebar.multiselect(
            "Select model categories",
            MODEL_CATEGORIES,
            default=['Deep Learning']
        )

        if 'Deep Learning' in model_categories:
            dl_params = get_dl_parameters()

            # Train models button
            if st.button("Train Models"):
                # Create preprocessor
                preprocessor = TimeSeriesPreprocessor()

                # Prepare datasets
                train_dataset, val_dataset, test_dataset, scaler = preprocessor.prepare_dl_datasets(
                    series=df[target_column],
                    sequence_length=dl_params['sequence_length'],
                    batch_size=dl_params['batch_size']
                )

                if train_dataset is not None:
                    results = train_models(
                        train_dataset,
                        val_dataset,
                        test_dataset,
                        dl_params,
                        target_column
                    )

                    # Display results
                    results_df = visualizer.plot_model_comparison(results)
                    if results_df is not None:
                        create_download_button(results_df)

def validate_and_prepare_dataframe(df, target_column):
    """Validate and prepare the dataframe for modeling."""
    try:
        # Ensure the target column exists
        if target_column not in df.columns:
            st.error(f"Target column '{target_column}' not found in the data.")
            return None

        # Convert target column to numeric
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            try:
                df[target_column] = pd.to_numeric(df[target_column])
            except Exception as e:
                st.error(f"Could not convert target column to numeric: {str(e)}")
                return None

        # Convert index to datetime if possible
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                st.warning(f"Could not convert index to datetime. Using numeric index instead: {str(e)}")
                df.index = pd.RangeIndex(start=0, stop=len(df))

        return df

    except Exception as e:
        st.error(f"Error preparing dataframe: {str(e)}")
        return None

def get_dl_parameters():
    """Get deep learning parameters from sidebar."""
    return {
        'sequence_length': st.sidebar.slider(
            "Sequence Length (Days)",
            10, 120,
            DEFAULT_SEQUENCE_LENGTH
        ),
        'epochs': st.sidebar.slider(
            "Number of Epochs",
            5, 50,
            DEFAULT_EPOCHS
        ),
        'batch_size': st.sidebar.slider(
            "Batch Size",
            16, 128,
            DEFAULT_BATCH_SIZE
        ),
        'learning_rate': st.sidebar.number_input(
            "Learning Rate",
            0.0001, 0.01,
            DEFAULT_LEARNING_RATE,
            format="%.4f"
        ),
        'rnn_units': st.sidebar.slider(
            "RNN/LSTM Units",
            16, 256,
            DEFAULT_RNN_UNITS
        ),
        'dropout_rate': st.sidebar.slider(
            "Dropout Rate",
            0.0, 0.5,
            DEFAULT_DROPOUT_RATE
        )
    }
    # After getting dl_params
    st.write("Debug: Deep Learning Parameters:", dl_params)
    st.write("Debug: Selected target column:", target_column)
    st.write("Debug: Target data type:", df[target_column].dtype)


def train_models(train_dataset, val_dataset, test_dataset, params, target_column):
    """Train selected models and return results."""
    results = []

    # Initialize models
    models = {
        'Simple RNN': SimpleRNNModel(
            name='Simple RNN',
            sequence_length=params['sequence_length']
        ),
        'LSTM': LSTMModel(
            name='LSTM',
            sequence_length=params['sequence_length']
        ),
        'Stacked LSTM+RNN': StackedModel(
            name='Stacked LSTM+RNN',
            sequence_length=params['sequence_length']
        )
    }

    # Train each model
    for model_name, model in models.items():
        with st.expander(f"**{model_name} Model**"):
            try:
                # Build and compile model
                if model_name == 'Stacked LSTM+RNN':
                    model.build(
                        lstm_units=params['rnn_units'] * 2,
                        rnn_units=params['rnn_units'],
                        dropout_rate=params['dropout_rate']
                    )
                else:
                    model.build(
                        units=params['rnn_units'],
                        dropout_rate=params['dropout_rate']
                    )

                model.model.compile(
                    optimizer=keras.optimizers.Adam(params['learning_rate']),
                    loss='mse',
                    metrics=['mae']
                )

                # Train and evaluate
                trainer = ModelTrainer(model, target_column)
                test_mae = trainer.train_and_evaluate(
                    train_dataset,
                    val_dataset,
                    test_dataset,
                    params['epochs']
                )

                if test_mae is not None:
                    results.append({
                        'Model': model_name,
                        'Test MAE': test_mae
                    })

            except Exception as e:
                st.error(f"Error training {model_name}: {str(e)}")
                continue

    return results


if __name__ == "__main__":
    main()
