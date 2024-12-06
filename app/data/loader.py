from config import (ERROR_MESSAGES, MIN_TRAINING_POINTS, 
                  GITHUB_RAW_BASE_URL, DEFAULT_EXAMPLE_FILE, 
                  TRADITIONAL_MODELS, ML_MODELS, DL_MODELS)
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Tuple, List, Dict
from pathlib import Path
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataLoader:
    """Class for handling data loading from various sources."""

    @staticmethod
    def validate_data(df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate loaded data meets basic requirements.

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            if df is None:
                return False, "No data loaded"

            if len(df) < MIN_TRAINING_POINTS:
                return False, f"Dataset too small. Minimum {MIN_TRAINING_POINTS} points required."

            # Check for excessive missing values (e.g., more than 50% in any column)
            missing_pct = df.isnull().mean()
            problem_cols = missing_pct[missing_pct > 0.5].index.tolist()
            if problem_cols:
                return False, f"Columns {problem_cols} have more than 50% missing values. Consider removing or imputing."

            return True, "Data validation successful"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def fix_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix data types and handle missing values for Arrow compatibility."""
        try:
            # Ensure datetime index if it's not already
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    pass

            # Fix column dtypes and handle missing values
            for col in df.columns:
                # Try to convert to numeric first
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Fill numeric missing values with forward fill, then backward fill
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                    continue
                except:
                    pass

                # Try to convert to datetime
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    # Fill datetime missing values with forward fill
                    df[col] = df[col].fillna(method='ffill')
                    continue
                except:
                    pass

                # If neither numeric nor datetime, convert to string and fill NaN
                df[col] = df[col].fillna('Unknown').astype(str)

            return df

        except Exception as e:
            st.error(f"Error fixing data types: {str(e)}")
            return df

    @classmethod
    def load_data(cls,
                  source_type: str,
                  uploaded_file: Optional[str] = None,
                  github_url: Optional[str] = None,
                  selected_file: Optional[str] = None,
                  ticker: Optional[str] = None,
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load data from various sources with comprehensive error handling.
        """
        try:
            st.write("Debug: Loading data from source type:", source_type)

            df = None

            # Define parser for date columns
            date_parser = lambda x: pd.to_datetime(x, errors='coerce')

            if source_type == "upload" and uploaded_file is not None:
                df = pd.read_csv(uploaded_file,
                                 parse_dates=True,
                                 index_col=0)
                df.index = pd.to_datetime(df.index)

            elif source_type == "github" and github_url and selected_file:
                df = pd.read_csv(selected_file,
                                 parse_dates=True,
                                 index_col=0)
                df.index = pd.to_datetime(df.index)

            elif source_type == "yfinance" and ticker:
                df = yf.download(ticker, start=start_date, end=end_date)

            else:
                return None

            # Convert any numeric columns to proper float type
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    st.warning("Could not convert index to datetime. Using default index.")
                    df.index = pd.RangeIndex(start=0, stop=len(df))

            # Sort index
            df.sort_index(inplace=True)

            # Create instance to call instance method
            loader = cls()
            # Fix data types
            df = loader.fix_dtypes(df)

            # Display data info for debugging
            st.write("Debug: DataFrame info after loading:")
            st.write(df.info())
            st.write("\nDebug: First few rows:")
            st.write(df.head())
            st.write("\nDebug: DataFrame dtypes:", df.dtypes)

            # Validate loaded data
            is_valid, message = cls.validate_data(df)
            if not is_valid:
                st.error(message)
                return None

            return df

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

    @staticmethod
    def get_github_files(repo_url: str) -> List[Tuple[str, str]]:
        """Fetch CSV files from a GitHub repository."""
        try:
            raw_base_url = repo_url.replace('github.com', 'raw.githubusercontent.com')
            raw_base_url = raw_base_url.replace('/tree/', '/')

            response = requests.get(repo_url)
            soup = BeautifulSoup(response.content, 'html.parser')

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

    def load_example_data(self, example_file=None):
        """Load example dataset from GitHub."""
        try:
            if example_file is None:
                example_file = DEFAULT_EXAMPLE_FILE

            # Construct raw GitHub URL
            file_url = f"{GITHUB_RAW_BASE_URL}/{example_file}"

            # Load the data
            df = pd.read_csv(file_url, parse_dates=True, index_col=0)

            # Ensure datetime index
            df.index = pd.to_datetime(df.index)

            # Sort index
            df.sort_index(inplace=True)

            # Fix data types
            df = self.fix_dtypes(df)

            return df

        except Exception as e:
            st.error(f"Error loading example data: {str(e)}")
            return None

    def prepare_data_for_model(self, df: pd.DataFrame, target_col: str, model_type: str) -> Optional[Dict]:
        """
        Prepare data for model training based on model type.

        Args:
            df: Input dataframe
            target_col: Target column name
            model_type: Type of model ('ARIMA', 'SARIMA', 'Random Forest', etc)

        Returns:
            Dict containing prepared data splits or None if error
        """
        try:
            # Validate input data
            if target_col not in df.columns:
                raise ValueError(f"Target column {target_col} not found in data")

            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    st.warning("Could not convert index to datetime. Using numeric index.")
                    df.index = pd.RangeIndex(start=0, stop=len(df))

            # Split data into train/test
            train_size = int(len(df) * 0.8)
            train_df = df[:train_size]
            test_df = df[train_size:]

            if model_type in ['ARIMA', 'SARIMA']:
                # For traditional models, return just the target series
                return {
                    'train': train_df[target_col],
                    'test': test_df[target_col],
                    'test_target': test_df[target_col],
                    'test_index': test_df.index
                }
            else:
                # For ML/DL models, create lagged features
                features = []
                lags = [1, 7, 14, 30]  # Example lag values

                for lag in lags:
                    df[f'lag_{lag}'] = df[target_col].shift(lag)

                # Add basic date features if index is datetime
                if isinstance(df.index, pd.DatetimeIndex):
                    df['month'] = df.index.month
                    df['day'] = df.index.day
                    df['day_of_week'] = df.index.dayofweek

                # Drop rows with NaN values from feature creation
                df = df.dropna()

                # Split features and target
                X = df.drop(columns=[target_col])
                y = df[target_col]

                # Re-split after feature creation
                train_size = int(len(df) * 0.8)
                X_train = X[:train_size]
                X_test = X[train_size:]
                y_train = y[:train_size]
                y_test = y[train_size:]

                return {
                    'train': (X_train, y_train),
                    'test': (X_test, y_test),
                    'test_target': y_test,
                    'test_index': y_test.index
                }

        except Exception as e:
            st.error(f"Error preparing data: {str(e)}")
            return None

    @staticmethod
    def get_dtypes_info(df: pd.DataFrame) -> pd.DataFrame:
        """Get comprehensive DataFrame information."""
        info_dict = {
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2),
            'Unique Values': df.nunique(),
            'Memory Usage': df.memory_usage(deep=True),
            'Sample Values': [df[col].iloc[0] if len(df) > 0 else None for col in df.columns]
        }
        return pd.DataFrame(info_dict)

    def prepare_model_data(self,
                         df: pd.DataFrame,
                         target_col: str,
                         model_type: str) -> Dict:
        """
        Prepare data specifically for each model type.
        Returns a dictionary with all necessary data splits and preprocessing info.
        """
        try:
            # Ensure target column exists
            if target_col not in df.columns:
                raise ValueError(f"Target column {target_col} not found in data")

            # Convert to datetime index if needed
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    st.warning("Could not convert index to datetime. Using numeric index.")
                    df.index = pd.RangeIndex(start=0, stop=len(df))

            # Create model-specific data preparation
            if model_type in TRADITIONAL_MODELS:
                return self._prepare_traditional_data(df, target_col)

            elif model_type in ML_MODELS:
                return self._prepare_ml_data(df, target_col)

            elif model_type in DL_MODELS:
                return self._prepare_dl_data(df, target_col)

            else:
                raise ValueError(f"Unknown model type: {model_type}")

        except Exception as e:
            st.error(f"Error preparing data: {str(e)}")
            return None

    def prepare_traditional_data(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.Series, pd.Series]:
        """
        Prepare data for traditional time series models (ARIMA, SARIMA etc.)
        
        Args:
            df: Input dataframe
            target_col: Name of target column
            
        Returns:
            Tuple containing training and test series
        """
        try:
            # Ensure target column exists
            if target_col not in df.columns:
                raise ValueError(f"Target column {target_col} not found in data")
                
            # Extract target series
            series = df[target_col]
            
            # Convert index to datetime if needed
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    series.index = pd.to_datetime(series.index)
                except:
                    st.warning("Could not convert index to datetime. Using numeric index.")
                    series.index = pd.RangeIndex(start=0, stop=len(series))

            # Split into train and test
            train_size = int(len(series) * 0.8)
            train_series = series[:train_size]
            test_series = series[train_size:]
            
            if len(train_series) < 2 or len(test_series) < 2:
                raise ValueError("Not enough data points after splitting")
                
            return train_series, test_series

        except Exception as e:
            st.error(f"Error preparing data for traditional models: {str(e)}")
            return None, None

    def _prepare_traditional_data(self, df: pd.DataFrame, target_col: str) -> Dict:
        """Internal method to prepare data for traditional models."""
        train_series, test_series = self.prepare_traditional_data(df, target_col)
        if train_series is not None and test_series is not None:
            return {
                'train': train_series,
                'test': test_series
            }
        return None

    def _prepare_ml_data(self, df: pd.DataFrame, target_col: str) -> Dict:
        """Prepare data for machine learning models."""
        # Implementation specific to ML models
        pass

    def _prepare_dl_data(self, df: pd.DataFrame, target_col: str) -> Dict:
        """Prepare data for deep learning models."""
        # Implementation specific to DL models
        pass