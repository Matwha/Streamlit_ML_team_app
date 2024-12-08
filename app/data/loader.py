import pandas as pd
import numpy as np
import streamlit as st
import time
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss


# In loader.py

class DataLoader:
    def __init__(self):
        self.df = None
        self.target_col = None
        self.index_col = None
        self.session_id = str(time.time())

    def load_data(self, file):
        """Load data and configure basic settings before any processing"""
        try:
            # Initial data load
            self.df = pd.read_csv(file)

            # First, handle unnamed columns
            unnamed_cols = [col for col in self.df.columns if 'Unnamed' in str(col)]
            if unnamed_cols:
                st.warning(f"Found {len(unnamed_cols)} unnamed columns")
                col1, col2 = st.columns(2)
                with col1:
                    action = st.radio(
                        "How to handle unnamed columns?",
                        ["Rename", "Drop"],
                        key=f"unnamed_action_{self.session_id}"
                    )

                if action == "Rename":
                    for col in unnamed_cols:
                        new_name = st.text_input(
                            f"New name for {col}",
                            value=f"col_{unnamed_cols.index(col)}",
                            key=f"rename_{col}_{self.session_id}"
                        )
                        self.df = self.df.rename(columns={col: new_name})
                else:
                    self.df = self.df.drop(columns=unnamed_cols)

                st.success("Unnamed columns handled")

            # Time Index Configuration (before any processing)
            st.subheader("Time Index Configuration")

            # First, identify potential date/time columns
            date_cols = [col for col in self.df.columns
                         if any(term in str(col).lower()
                                for term in ['date', 'time', 'month', 'year'])]

            # Allow manual selection if no date columns found
            if not date_cols:
                date_cols = self.df.columns.tolist()
                st.warning("No date/time columns automatically detected")

            index_col = st.selectbox(
                "Select Time Index Column",
                date_cols,
                key=f"index_select_{self.session_id}"
            )

            # Period Index Configuration
            st.write("### Configure Period Index")
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
                    key=f"freq_select_{self.session_id}"
                )

            with col2:
                if freq in ['W', 'M', 'Q']:
                    period_anchor = st.selectbox(
                        "Period Anchor",
                        ['Start', 'End'],
                        key=f"anchor_select_{self.session_id}"
                    )

            # Apply Period Index
            if st.button("Apply Period Index", key=f"apply_period_{self.session_id}"):
                try:
                    # Convert to datetime
                    self.df.index = pd.to_datetime(self.df[index_col])

                    # Apply period anchor if selected
                    if freq in ['W', 'M', 'Q'] and period_anchor == 'End':
                        offset_map = {'W': 'W-SAT', 'M': 'M', 'Q': 'Q'}
                        self.df.index = self.df.index + pd.offsets.to_offset(offset_map[freq])

                    # Convert to period index
                    self.df.index = self.df.index.to_period(freq)
                    self.df = self.df.drop(columns=[index_col])

                    # Verify index type
                    if isinstance(self.df.index, pd.PeriodIndex):
                        st.success("Period Index configured successfully!")
                        # Show index details
                        st.write("Index Details:")
                        st.write({
                            'Type': type(self.df.index),
                            'Frequency': self.df.index.freq,
                            'Range': f"{self.df.index[0]} to {self.df.index[-1]}"
                        })
                    else:
                        st.error("Failed to create PeriodIndex")

                except Exception as e:
                    st.error(f"Error configuring period index: {str(e)}")
                    return None

            # Column Management
            if st.checkbox("Show Column Management", key=f"show_col_mgmt_{self.session_id}"):
                st.write("### Column Management")

                # Column selection
                selected_cols = st.multiselect(
                    "Select and Reorder Columns",
                    self.df.columns.tolist(),
                    default=self.df.columns.tolist(),
                    key=f"col_select_{self.session_id}"
                )

                if selected_cols:
                    self.df = self.df[selected_cols]

                # Column renaming
                if st.checkbox("Rename Columns", key=f"show_rename_{self.session_id}"):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        col_to_rename = st.selectbox(
                            "Select Column",
                            self.df.columns.tolist(),
                            key=f"rename_select_{self.session_id}"
                        )
                    with col2:
                        new_name = st.text_input(
                            "New Name",
                            value=col_to_rename,
                            key=f"new_name_{self.session_id}"
                        )
                    with col3:
                        if st.button("Rename", key=f"rename_btn_{self.session_id}"):
                            self.df = self.df.rename(columns={col_to_rename: new_name})
                            st.success(f"Renamed {col_to_rename} to {new_name}")

            # Show initial data preview
            st.write("### Data Preview")
            st.dataframe(self.df.head())

            return self.df

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

    def create_lagged_features(self, data, seq_length):
        """Create lagged features for a given sequence length"""
        df = pd.DataFrame(data)
        for i in range(1, seq_length + 1):
            df[f'lag_{i}'] = df.iloc[:, 0].shift(i)
        return df.dropna()

    def verify_data_ready(self, target_col=None):
        """Verify data is ready for modeling"""
        if self.df is None:
            st.error("No data available for processing")
            return False

        if self.df.empty:
            st.error("DataFrame is empty")
            return False

        if target_col and target_col not in self.df.columns:
            st.error(f"Target column '{target_col}' not found in data")
            st.write("Available columns:", self.df.columns.tolist())
            return False

        if not isinstance(self.df.index, pd.PeriodIndex):
            st.error("Period Index not properly configured")
            st.write("Current index type:", type(self.df.index))
            return False

        return True

    def debug_data_state(self, message="Current Data State"):
        """Helper function to display data state for debugging"""
        with st.expander("Debug Information"):
            st.write(f"DEBUG - {message}")
            if self.df is not None:
                st.write("Shape:", self.df.shape)
                st.write("Index Type:", type(self.df.index))
                st.write("Data Types:", self.df.dtypes)
                st.write("Sample:", self.df.head())