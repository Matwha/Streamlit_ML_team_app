import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
import os
import requests
from bs4 import BeautifulSoup
import io

# Suppress warnings and TensorFlow logs
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def get_github_files(repo_url):
    """
    Fetch CSV files from a GitHub repository
    """
    try:
        # Convert tree URL to raw content URL format
        raw_base_url = repo_url.replace('github.com', 'raw.githubusercontent.com').replace('/tree/', '/')
        
        # Get the HTML content
        response = requests.get(repo_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all links that end with .csv
        csv_files = []
        for link in soup.find_all('a'):
            href = link.get('href', '')
            if href.endswith('.csv'):
                file_name = href.split('/')[-1]
                raw_url = f"{raw_base_url}/{file_name}"
                csv_files.append((file_name, raw_url))
        
        return csv_files
    except Exception as e:
        st.error(f"Error fetching GitHub files: {str(e)}")
        return []
    
def load_data(source_type, uploaded_file=None, github_url=None, selected_file=None):
    """
    Load data from various sources with proper error handling
    """
    try:
        if source_type == "upload" and uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("Successfully loaded uploaded data!")
            return df
            
        elif source_type == "github" and github_url and selected_file:
            df = pd.read_csv(selected_file)
            st.success("Successfully loaded GitHub data!")
            return df
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None
    
def explore_data(df):
    """
    Perform basic data exploration
    """
    st.header("Data Exploration")
    
    # Basic information
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Number of rows:", df.shape[0])
    with col2:
        st.write("Number of columns:", df.shape[1])
    with col3:
        st.write("Memory usage:", f"{df.memory_usage().sum() / 1024**2:.2f} MB")

    # Show first few rows
    st.subheader("First Few Rows")
    st.dataframe(df.head())
    
    # Data types information
    st.subheader("Data Types Information")
    st.dataframe(pd.DataFrame({
        'Data Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum()
    }))

    # Basic statistics
    st.subheader("Numerical Columns Statistics")
    st.dataframe(df.describe())

    # Time series utils
    st.subheader("Time Series Visualization")
    if isinstance(df.index, pd.DatetimeIndex):
        time_index = df.index
    else:
        try:
            # Try to convert the first column to datetime if it looks like dates
            first_col = df.iloc[:, 0]
            if pd.to_datetime(first_col, errors='coerce').notnull().all():
                time_index = pd.to_datetime(first_col)
            else:
                time_index = np.arange(len(df))
        except:
            time_index = np.arange(len(df))

    # Plot numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        selected_cols = st.multiselect(
            "Select columns to visualize",
            numeric_cols,
            default=numeric_cols[0]
        )
        
        if selected_cols:
            fig, ax = plt.subplots(figsize=(12, 6))
            for col in selected_cols:
                ax.plot(time_index, df[col], label=col)
            ax.set_title("Time Series Plot")
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()

            # Correlation heatmap
            if len(selected_cols) > 1:
                st.subheader("Correlation Heatmap")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(df[selected_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
                plt.close()

@st.cache_resource
def build_rnn_model(sequence_length, units, dropout_rate):
    """Build RNN model with caching"""
    try:
        inputs = keras.Input(shape=(sequence_length, 1))
        x = layers.SimpleRNN(units, dropout=dropout_rate)(inputs)
        outputs = layers.Dense(1)(x)
        model = keras.Model(inputs, outputs)
        return model
    except Exception as e:
        st.error(f"Error building RNN model: {str(e)}")
        return None

@st.cache_resource
def build_lstm_model(sequence_length, units, dropout_rate):
    """Build LSTM model with caching"""
    try:
        inputs = keras.Input(shape=(sequence_length, 1))
        x = layers.LSTM(units, dropout=dropout_rate)(inputs)
        outputs = layers.Dense(1)(x)
        model = keras.Model(inputs, outputs)
        return model
    except Exception as e:
        st.error(f"Error building LSTM model: {str(e)}")
        return None

@st.cache_resource
def build_stacked_model(sequence_length, lstm_units, rnn_units, dropout_rate):
    """Build stacked LSTM+RNN model with caching"""
    try:
        inputs = keras.Input(shape=(sequence_length, 1))
        x = layers.LSTM(lstm_units, dropout=dropout_rate, return_sequences=True)(inputs)
        x = layers.SimpleRNN(rnn_units, dropout=dropout_rate)(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1)(x)
        model = keras.Model(inputs, outputs)
        return model
    except Exception as e:
        st.error(f"Error building stacked model: {str(e)}")
        return None

# Data preprocessing functions
@st.cache_data
def prepare_data(series):
    """Prepare the raw data with better error handling"""
    try:
        # Convert series to numpy array if it's not already
        if isinstance(series, pd.Series):
            data = series.values
        else:
            data = np.array(series)
            
        # Ensure data is 2D
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
            
        # Create log returns
        log_returns = np.log(data[1:] / data[:-1])
        
        # Handle any infinite values
        log_returns = np.nan_to_num(log_returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(log_returns)
        
        return scaled_data, scaler
        
    except Exception as e:
        st.error(f"Error in data preparation: {str(e)}")
        return None, None

def create_datasets(scaled_data, sequence_length, batch_size, val_split=0.2, test_split=0.2):
    """Create TensorFlow datasets with proper sizing based on sequences_count"""
    try:
        n = len(scaled_data)
        sequences_count = n - sequence_length

        # Ensure enough data points
        if sequences_count < 3:
            st.error(f"Not enough sequences for the given sequence length ({sequence_length}). Minimum required is 3 sequences, but got {sequences_count}.")
            return None, None, None

        # Calculate the effective sizes based on sequences_count
        train_size = int(sequences_count * (1 - val_split - test_split))
        val_size = int(sequences_count * val_split)
        test_size = sequences_count - train_size - val_size

        # Adjust proportions if any split is too small
        min_split_size = 1  # At least one sequence per split
        if train_size < min_split_size or val_size < min_split_size or test_size < min_split_size:
            st.warning("Adjusting dataset splits to ensure all splits have at least one sample.")
            train_size = max(min_split_size, int(sequences_count * 0.7))
            val_size = max(min_split_size, int(sequences_count * 0.15))
            test_size = sequences_count - train_size - val_size

        # Prepare sequences and targets
        sequences = []
        targets = []
        for i in range(sequences_count):
            sequences.append(scaled_data[i:(i + sequence_length)])
            targets.append(scaled_data[i + sequence_length])

        sequences = np.array(sequences)
        targets = np.array(targets)

        # Split the sequences and targets
        train_sequences = sequences[:train_size]
        train_targets = targets[:train_size]

        val_sequences = sequences[train_size:train_size + val_size]
        val_targets = targets[train_size:train_size + val_size]

        test_sequences = sequences[train_size + val_size:]
        test_targets = targets[train_size + val_size:]

        # Log dataset sizes for debugging
        st.write(f"Total data points: {n}")
        st.write(f"Sequence length: {sequence_length}")
        st.write(f"Total sequences: {sequences_count}")
        st.write(f"Train size: {len(train_sequences)}, Validation size: {len(val_sequences)}, Test size: {len(test_sequences)}")
        st.write("Train sequences shape:", train_sequences.shape)
        st.write("Validation sequences shape:", val_sequences.shape)
        st.write("Test sequences shape:", test_sequences.shape)

        # Handle cases where any dataset split is empty
        if len(train_sequences) == 0 or len(val_sequences) == 0 or len(test_sequences) == 0:
            st.error("One of the datasets (train, val, or test) is empty. Adjust the sequence length or split ratios.")
            return None, None, None

        # Adjust batch size based on train_size
        adjusted_batch_size = max(1, min(batch_size, train_size // 10))

        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_targets)).shuffle(1024).batch(adjusted_batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((val_sequences, val_targets)).batch(adjusted_batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_sequences, test_targets)).batch(adjusted_batch_size)

        return train_dataset, val_dataset, test_dataset

    except Exception as e:
        st.error(f"Error creating datasets: {str(e)}")
        return None, None, None

def prepare_dl_datasets(series, sequence_length, batch_size, val_split=0.2, test_split=0.2):
    """Main function to prepare data with validation"""
    # Check for NaN values
    if series.isna().any():
        st.error("Data contains missing values. Please clean the data first.")
        return None, None, None, None
        
    # Check if series is too short
    if len(series) < sequence_length + 3:  # At least 3 sequences
        st.error(f"Time series too short. Need at least {sequence_length + 3} points.")
        return None, None, None, None
        
    try:
        # Create progress containers
        prep_status = st.empty()
        prep_progress = st.progress(0)

        # Data preparation steps
        prep_status.text("Preparing data...")
        prep_progress.progress(0.2)

        scaled_data, scaler = prepare_data(series)
        if scaled_data is None:
            prep_progress.empty()
            prep_status.empty()
            return None, None, None, None
        prep_progress.progress(0.6)

        prep_status.text("Creating datasets...")
        train_dataset, val_dataset, test_dataset = create_datasets(
            scaled_data=scaled_data,
            sequence_length=sequence_length,
            batch_size=batch_size,
            val_split=val_split,
            test_split=test_split
        )

        if train_dataset is None:
            st.error("Failed to create datasets. Please adjust the parameters.")
            prep_progress.empty()
            prep_status.empty()
            return None, None, None, None

        prep_progress.progress(1.0)
        prep_status.text("Data preparation complete!")

        # Clean up progress indicators
        prep_status.empty()
        prep_progress.empty()

        return train_dataset, val_dataset, test_dataset, scaler

    except Exception as e:
        st.error(f"Error in data preparation: {str(e)}")
        return None, None, None, None

def train_and_evaluate_model(model_name, model, train_dataset, val_dataset, test_dataset, epochs, target_column):
    """Train and evaluate with better error handling"""
    try:
        # Create progress containers
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_container = st.empty()
        
        # Custom callback
        class CustomCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(f"Training epoch {epoch + 1}/{epochs}")
                if logs:
                    metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
                    metrics_container.text(f"Current metrics: {metrics_str}")
                    
        callbacks = [
            CustomCallback(),
            keras.callbacks.ModelCheckpoint(
                f"{target_column}_{model_name.lower().replace(' ', '_')}.keras",
                save_best_only=True,
                monitor='loss',
                mode='min'
            ),
            keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=5,
                restore_best_weights=True
            )
        ]
        
        # Train model
        history = model.fit(
            train_dataset,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=val_dataset,
            verbose=0
        )
        
        # Evaluate
        test_loss = model.evaluate(test_dataset, verbose=0)
        test_mae = test_loss[1] if isinstance(test_loss, list) else test_loss
        
        # Plot history
        if hasattr(history, 'history'):
            fig, ax = plt.subplots()
            ax.plot(history.history['loss'], label='Training Loss')
            if 'val_loss' in history.history:
                ax.plot(history.history['val_loss'], label='Validation Loss')
            ax.set_title(f'{model_name} Training History')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
        
        # Clean up
        progress_bar.empty()
        status_text.empty()
        metrics_container.empty()
        
        return test_mae
        
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        return None

def main():
    """Main application function with improved error handling and data source options"""
    st.set_page_config(layout="wide", page_title="Time Series Forecasting App")

    try:
        # Title and description
        st.title("Time Series Forecasting Application")
        st.markdown("""
        This app performs univariate time series forecasting using various models including deep learning.
        Models available:
        - **Traditional Models:** ARIMA, SARIMA
        - **Machine Learning:** Random Forest, XGBoost
        - **Deep Learning:** RNN, LSTM, Stacked LSTM+RNN
        """)

        # Sidebar - Data Source Selection
        st.sidebar.header("Data Source")
        data_source = st.sidebar.radio(
            "Select data source",
            ["GitHub Repository", "Upload File", "Example Data"]
        )

        df = None
        if data_source == "GitHub Repository":
            github_url = st.sidebar.text_input(
                "GitHub repository URL",
                "https://github.com/PJalgotrader/Deep_forecasting-USU/tree/main/data",
            )
            
            if github_url:
                csv_files = get_github_files(github_url)
                if csv_files:
                    selected_file = st.sidebar.selectbox(
                        "Select a CSV file",
                        [file[0] for file in csv_files],
                    )
                    if selected_file:
                        file_url = next(file[1] for file in csv_files if file[0] == selected_file)
                        df = load_data("github", github_url=github_url, selected_file=file_url)

        elif data_source == "Upload File":
            uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
            if uploaded_file:
                df = load_data("upload", uploaded_file=uploaded_file)
                
        else:  # Example Data
            try:
                df = pd.read_csv("https://raw.githubusercontent.com/PJalgotrader/Deep_forecasting-USU/main/data/yfinance.csv", 
                               index_col=0, header=[0,1])['Close']
                st.sidebar.info("Using example GLD stock price data")
            except Exception as e:
                st.sidebar.error(f"Error loading example data: {str(e)}")
                return

        # Data validation
        if df is not None:
            # Perform data exploration first
            explore_data(df)
            
            if isinstance(df, pd.DataFrame):
                num_rows = df.shape[0]
            else:
                num_rows = len(df)

            if num_rows < 100:  # arbitrary minimum
                st.error("Dataset too small for reliable training (minimum 100 points required)")
                return

            # Target variable selection
            target_column = st.sidebar.selectbox(
                "Select target variable",
                df.columns if isinstance(df, pd.DataFrame) else ['GLD']
            )

            # Model Selection
            st.sidebar.header("Model Selection")
            model_categories = st.sidebar.multiselect(
                "Select model categories",
                ['Traditional', 'Machine Learning', 'Deep Learning'],
                default=['Deep Learning']
            )

            # Get selected models
            models_to_run = []
            if 'Traditional' in model_categories:
                traditional_models = st.sidebar.multiselect(
                    "Select traditional models",
                    ['ARIMA', 'SARIMA'],
                    default=['ARIMA']
                )
                models_to_run.extend(traditional_models)

            if 'Machine Learning' in model_categories:
                ml_models = st.sidebar.multiselect(
                    "Select ML models",
                    ['Random Forest', 'XGBoost'],
                    default=['Random Forest']
                )
                models_to_run.extend(ml_models)

            if 'Deep Learning' in model_categories:
                dl_models = st.sidebar.multiselect(
                    "Select deep learning models",
                    ['Simple RNN', 'LSTM', 'Stacked LSTM+RNN'],
                    default=['Simple RNN', 'LSTM']
                )
                models_to_run.extend(dl_models)

            # Model parameters for deep learning
            dl_params = {}
            if any(model in models_to_run for model in ['Simple RNN', 'LSTM', 'Stacked LSTM+RNN']):
                st.sidebar.header("Deep Learning Parameters")
                sequence_length = st.sidebar.slider("Sequence Length (Days)", 10, 120, 60)
                epochs = st.sidebar.slider("Number of Epochs", 5, 50, 20)
                batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)
                learning_rate = st.sidebar.number_input("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
                rnn_units = st.sidebar.slider("RNN/LSTM Units", 16, 256, 64)
                dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.1)
                
                dl_params = {
                    'sequence_length': sequence_length,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'rnn_units': rnn_units,
                    'dropout_rate': dropout_rate
                }
                
                if 'Stacked LSTM+RNN' in models_to_run:
                    stacked_units = st.sidebar.slider("Stacked Layer Units", 32, 256, 128)
                    dl_params['stacked_units'] = stacked_units

                # Additional splits customization
                st.sidebar.header("Dataset Splits")
                val_split = st.sidebar.slider("Validation Split (%)", 5, 30, 20, step=1) / 100.0
                test_split = st.sidebar.slider("Test Split (%)", 5, 30, 20, step=1) / 100.0

                # Ensure that val_split + test_split < 1
                if val_split + test_split >= 0.9:
                    st.sidebar.warning("Validation and Test splits are too large. Adjusting to fit the data.")
                    val_split = 0.2
                    test_split = 0.2
                    st.sidebar.write("Validation Split set to 20% and Test Split set to 20%.")

            # Train models button
            if st.button("Train Models"):
                results = []
                
                # Prepare data for deep learning models
                if any(model in models_to_run for model in ['Simple RNN', 'LSTM', 'Stacked LSTM+RNN']):
                    try:
                        with st.spinner('Preparing datasets...'):
                            train_dataset, val_dataset, test_dataset, scaler = prepare_dl_datasets(
                                series=df[target_column], 
                                sequence_length=dl_params['sequence_length'], 
                                batch_size=dl_params['batch_size'],
                                val_split=val_split,
                                test_split=test_split
                            )
                            
                            if train_dataset is None:
                                st.error("Failed to prepare datasets")
                                return

                        # Training models
                        for model_name in models_to_run:
                            if model_name not in ['Simple RNN', 'LSTM', 'Stacked LSTM+RNN']:
                                continue  # Skip non-DL models here

                            with st.expander(f"**{model_name} Model**"):
                                try:
                                    model = None
                                    if model_name == 'Simple RNN':
                                        model = build_rnn_model(
                                            sequence_length=dl_params['sequence_length'], 
                                            units=dl_params['rnn_units'], 
                                            dropout_rate=dl_params['dropout_rate']
                                        )
                                    elif model_name == 'LSTM':
                                        model = build_lstm_model(
                                            sequence_length=dl_params['sequence_length'], 
                                            units=dl_params['rnn_units'], 
                                            dropout_rate=dl_params['dropout_rate']
                                        )
                                    elif model_name == 'Stacked LSTM+RNN':
                                        model = build_stacked_model(
                                            sequence_length=dl_params['sequence_length'], 
                                            lstm_units=dl_params['stacked_units'], 
                                            rnn_units=dl_params['rnn_units'], 
                                            dropout_rate=dl_params['dropout_rate']
                                        )

                                    if model is not None:
                                        model.compile(
                                            optimizer=keras.optimizers.Adam(dl_params['learning_rate']),
                                            loss='mse',
                                            metrics=['mae']
                                        )
                                        
                                        test_mae = train_and_evaluate_model(
                                            model_name=model_name, 
                                            model=model, 
                                            train_dataset=train_dataset, 
                                            val_dataset=val_dataset, 
                                            test_dataset=test_dataset, 
                                            epochs=dl_params['epochs'], 
                                            target_column=target_column
                                        )
                                        
                                        if test_mae is not None:
                                            results.append({'Model': model_name, 'Test MAE': test_mae})
                                        
                                except Exception as e:
                                    st.error(f"Error training {model_name}: {str(e)}")
                                    continue
                                    
                    except Exception as e:
                        st.error(f"Error in data preparation pipeline: {str(e)}")
                        return

                # Display results comparison
                if results:
                    st.header("Model Comparison")
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df.style.highlight_min(axis=0))
                    
                    # Add download button
                    st.download_button(
                        label="Download results as CSV",
                        data=results_df.to_csv(index=False).encode('utf-8'),
                        file_name='forecast_results.csv',
                        mime='text/csv',
                    )
        else:
            st.warning("Please select a data source and load data to continue")

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.write("Please refresh the page and try again.")

if __name__ == "__main__":
    main()
