"""Data loading functions for different data sources."""


import requests
from bs4 import BeautifulSoup

import pandas as pd
import streamlit as st


def fix_dtypes(df):
    """Fix data types for Arrow compatibility."""
    for col in df.columns:
        # Get column dtype
        dtype = df[col].dtype

        # Fix object dtypes
        if dtype == 'object':
            # Try to convert to datetime
            try:
                df[col] = pd.to_datetime(df[col])
                continue
            except:
                pass

            # Try to convert to numeric
            try:
                df[col] = pd.to_numeric(df[col])
                continue
            except:
                pass

            # If still object type, convert to string
            df[col] = df[col].astype(str)

    return df


def load_data(source_type, uploaded_file=None, github_url=None, selected_file=None):
    """Load data with proper type handling."""
    try:
        st.write("Debug: Loading data from source type:", source_type)

        if source_type == "upload" and uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        elif source_type == "github" and github_url and selected_file:
            df = pd.read_csv(selected_file)
        else:
            return None

        # Handle date columns
        for col in df.columns:
            # Try to convert to datetime if column name contains 'date' or 'time'
            if any(word in col.lower() for word in ['date', 'time']):
                try:
                    df[col] = pd.to_datetime(df[col])
                    st.write(f"Debug: Converted {col} to datetime")
                except:
                    st.write(f"Debug: Failed to convert {col} to datetime")
                    pass

        df = fix_dtypes(df)
        st.write("Debug: DataFrame dtypes after loading:", df.dtypes)
        return df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def get_dtypes_info(df):
    """Get DataFrame data types information."""
    dtypes_info = pd.DataFrame({
        'Data Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Sample Values': [df[col].iloc[0] if len(df) > 0 else None for col in df.columns]
    })
    return dtypes_info

def get_github_files(repo_url):
    """
    Fetch CSV files from a GitHub repository.

    Args:
        repo_url (str): URL of the GitHub repository

    Returns:
        list: List of tuples containing (filename, raw_url)
    """
    try:
        raw_base_url = repo_url.replace('github.com', 'raw.githubusercontent.com').replace('/tree/', '/')
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


def load_example_data(url):
    """
    Load example dataset from URL.

    Args:
        url (str): URL of the example dataset

    Returns:
        pd.DataFrame: Loaded dataframe or None if error
    """
    try:
        df = pd.read_csv(url, index_col=0, header=[0, 1])['Close']
        return df
    except Exception as e:
        st.error(f"Error loading example data: {str(e)}")
        return None