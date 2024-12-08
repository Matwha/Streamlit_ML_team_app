# In main.py:

import streamlit as st
import pandas as pd
import numpy as np
from config import *
from data.loader import DataLoader
from models.evaluation import ModelEvaluator, plot_model_comparison, plot_predictions
from models.arima import ARIMAModel, SARIMAModel
from models.rf_sgb import RandomForestModel, XGBoostModel
from models.rnn_lstm import SimpleRNNModel, LSTMModel, StackedModel

def initialize_model(model_name, config):
    """Initialize model with given configuration"""
    if model_name == 'LSTM':
        model = LSTMModel(
            sequence_length=config['sequence_length'],
            n_features=1
        )
        model.build(units=config['units'])
    elif model_name == 'Simple RNN':
        model = SimpleRNNModel(
            sequence_length=config['sequence_length'],
            n_features=1
        )
        model.build(units=config['units'])
    elif model_name == 'Stacked LSTM+RNN':
        model = StackedModel(
            sequence_length=config['sequence_length'],
            n_features=1
        )
        model.build(lstm_units=config['units'])
    elif model_name == 'ARIMA':
        model = ARIMAModel(order=(config['p'], config['d'], config['q']))
    elif model_name == 'SARIMA':
        model = SARIMAModel(
            order=(config['p'], config['d'], config['q']),
            seasonal_order=(config['p'], config['d'], config['q'], config['s'])
        )
    elif model_name == 'Random Forest':
        model = RandomForestModel(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth']
        )
    else:  # XGBoost
        model = XGBoostModel(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth']
        )
    return model

def get_model_preprocessing_options(self, model_name):
    """Get model-specific preprocessing options"""
    options = {}
    if model_name in ['Simple RNN', 'LSTM', 'Stacked LSTM+RNN']:
        options.update({
            'sequence_scaler': st.selectbox(
                f"{model_name} Sequence Scaler",
                ['StandardScaler', 'MinMaxScaler', 'None'],
                key=f"seq_scaler_{model_name}"
            ),
            'create_lags': st.checkbox(
                f"{model_name} Create Lagged Features",
                value=True,
                key=f"create_lags_{model_name}"
            ),
            'lag_length': st.number_input(
                f"{model_name} Lag Length",
                min_value=1, max_value=20, value=12,
                key=f"lag_length_{model_name}"
            )
        })
    return options


def initialize_app_state():
    """Initialize application state and session variables."""
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
        st.session_state.session_id = str(int(time.time()))

    if 'data_state' not in st.session_state:
        st.session_state.data_state = {
            'raw_data': None,
            'processed_data': None,
            'file_hash': None,
            'index_col': None,
            'target_col': None
        }


def create_sidebar():
    """Create and handle sidebar elements."""
    with st.sidebar:
        selected_metrics = st.multiselect(
            "Comparison Metrics",
            options=AVAILABLE_METRICS,
            default=DEFAULT_METRICS,
            key=f"metrics_select_{st.session_state.session_id}"
        )

        selected_models = st.multiselect(
            "Select Models (max 3)",
            options=ALL_MODELS,
            max_selections=3,
            key=f"models_select_{st.session_state.session_id}"
        )

        return selected_metrics, selected_models


def handle_data_upload():
    """Handle data upload and initial processing."""
    uploaded_file = st.file_uploader(
        "Upload Dataset (CSV)",
        type=['csv'],
        key=f"file_uploader_{st.session_state.session_id}"
    )

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.data_state['raw_data'] = df
            st.session_state.data_state['file_hash'] = hash(uploaded_file.name)
            return True
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return False
    return False


def main():
    st.set_page_config(layout="wide", page_title=APP_TITLE)
    st.title(APP_TITLE)

    # Initialize app state
    initialize_app_state()

    # Create sidebar and get selections
    selected_metrics, selected_models = create_sidebar()

    # Create main tabs
    main_tabs = st.tabs([
        "Data & Analysis",
        "Model Configuration",
        "Training Monitor",
        "Results & Comparison"
    ])

    # Handle Data & Analysis Tab
    with main_tabs[0]:
        if handle_data_upload():
            # Continue with data processing...
            pass


def handle_unnamed_columns(df, file_hash):
    """Handle unnamed columns in the dataset."""
    unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
    if unnamed_cols:
        st.warning(f"Found {len(unnamed_cols)} unnamed columns")

        handle_unnamed = st.radio(
            "How to handle unnamed columns?",
            ["Rename", "Drop"],
            key=f"unnamed_action_{file_hash}"
        )

        if handle_unnamed == "Rename":
            col_renames = {}
            for i, col in enumerate(unnamed_cols):
                new_name = st.text_input(
                    f"New name for {col}",
                    value=f"col_{i}",
                    key=f"rename_unnamed_{i}_{file_hash}"
                )
                col_renames[col] = new_name

            if st.button("Apply Column Renaming", key=f"apply_rename_{file_hash}"):
                df = df.rename(columns=col_renames)
                st.success("Columns renamed successfully!")
        else:
            if st.button("Drop Unnamed Columns", key=f"drop_unnamed_{file_hash}"):
                df = df.drop(columns=unnamed_cols)
                st.success("Unnamed columns dropped!")

    return df


def configure_time_index(df, file_hash):
    """Configure time index for the dataset."""
    st.write("### Time Index Configuration")

    date_cols = [col for col in df.columns
                 if any(term in str(col).lower()
                        for term in ['date', 'time', 'month', 'year'])]

    if not date_cols:
        date_cols = df.columns.tolist()
        st.warning("No date/time columns automatically detected")

    index_col = st.selectbox(
        "Select Time Index Column",
        date_cols,
        key=f"time_index_select_{file_hash}"
    )

    if index_col:
        freq_options = {
            'B': 'Business Day',
            'D': 'Calendar Day',
            'W': 'Weekly',
            'M': 'Monthly',
            'Q': 'Quarterly',
            'Y': 'Yearly'
        }

        col1, col2 = st.columns(2)
        with col1:
            freq = st.selectbox(
                "Select Frequency",
                options=list(freq_options.keys()),
                format_func=lambda x: freq_options[x],
                key=f"freq_select_{file_hash}"
            )

        with col2:
            period_anchor = None
            if freq in ['W', 'M', 'Q']:
                period_anchor = st.selectbox(
                    "Period Anchor",
                    ['Start', 'End'],
                    key=f"anchor_select_{file_hash}"
                )

        if st.button("Apply Period Index", key=f"apply_period_{file_hash}"):
            try:
                df = convert_to_period_index(df, index_col, freq, period_anchor)
                st.success("Period Index configured successfully!")
                return df, index_col
            except Exception as e:
                st.error(f"Error configuring period index: {str(e)}")

    return df, None


def convert_to_period_index(df, index_col, freq, period_anchor=None):
    """Convert dataframe index to period index."""
    df = df.copy()
    df.index = pd.to_datetime(df[index_col])

    if freq in ['W', 'M', 'Q'] and period_anchor == 'End':
        offset_map = {'W': 'W-SAT', 'M': 'M', 'Q': 'Q'}
        df.index = df.index + pd.offsets.to_offset(offset_map[freq])

    df.index = df.index.to_period(freq)
    df = df.drop(columns=[index_col])

    return df


def handle_column_management(df, file_hash):
    """Handle column selection and renaming."""
    if st.checkbox("Show Column Management", key=f"show_col_mgmt_{file_hash}"):
        st.write("### Column Management")

        selected_cols = st.multiselect(
            "Select and Reorder Columns",
            df.columns.tolist(),
            default=df.columns.tolist(),
            key=f"col_select_{file_hash}"
        )

        if selected_cols:
            df = df[selected_cols]

        if st.checkbox("Rename Columns", key=f"show_rename_{file_hash}"):
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                col_to_rename = st.selectbox(
                    "Select Column",
                    df.columns.tolist(),
                    key=f"rename_select_{file_hash}"
                )
            with col2:
                new_name = st.text_input(
                    "New Name",
                    value=col_to_rename,
                    key=f"new_name_{file_hash}"
                )
            with col3:
                if st.button("Rename", key=f"rename_btn_{file_hash}"):
                    df = df.rename(columns={col_to_rename: new_name})
                    st.success(f"Renamed {col_to_rename} to {new_name}")

    return df



if __name__ == "__main__":
    main()