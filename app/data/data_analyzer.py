"""Enhanced time series data analysis module."""

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import streamlit as st
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class TimeSeriesAnalyzer:
    """Class for comprehensive time series analysis."""

    def __init__(self, df: pd.DataFrame):
        """Initialize with a DataFrame."""
        self.df = df
        self.analysis_results = {}

    def analyze_basic_stats(self, column: str) -> Dict:
        """Calculate basic statistical measures."""
        series = self.df[column]
        
        # First check if the series is datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return {
                'Data Type': 'datetime',
                'Is Numeric': False,
                'Start Date': series.min().strftime('%Y-%m-%d'),
                'End Date': series.max().strftime('%Y-%m-%d'),
                'Date Range (days)': (series.max() - series.min()).days,
                'Number of Unique Dates': len(series.unique()),
                'Missing Values': series.isnull().sum(),
                'Note': 'Datetime column. Statistical analysis will be performed on numeric representations of dates.'
            }
        
        # Try to convert to datetime if it looks like dates
        try:
            if isinstance(series.iloc[0], str):
                datetime_series = pd.to_datetime(series, errors='raise')
                return {
                    'Data Type': 'datetime (converted)',
                    'Is Numeric': False,
                    'Start Date': datetime_series.min().strftime('%Y-%m-%d'),
                    'End Date': datetime_series.max().strftime('%Y-%m-%d'),
                    'Date Range (days)': (datetime_series.max() - datetime_series.min()).days,
                    'Number of Unique Dates': len(datetime_series.unique()),
                    'Missing Values': datetime_series.isnull().sum(),
                    'Note': 'String dates converted to datetime. Statistical analysis will be performed on numeric representations of dates.'
                }
        except (ValueError, TypeError, pd.errors.ParserError):
            pass
        
        # Check if the series is numeric
        if pd.api.types.is_numeric_dtype(series):
            stats_dict = {
                'Data Type': str(series.dtype),
                'Is Numeric': True,
                'Mean': float(series.mean()),
                'Median': float(series.median()),
                'Std Dev': float(series.std()),
                'Min': float(series.min()),
                'Max': float(series.max()),
                'Skewness': float(stats.skew(series.dropna())),
                'Kurtosis': float(stats.kurtosis(series.dropna())),
                'Missing Values': int(series.isnull().sum()),
            }
            
            # Add percentiles as individual values rather than a dict
            stats_dict['Percentile_25'] = float(series.quantile(0.25))
            stats_dict['Percentile_50'] = float(series.quantile(0.50))
            stats_dict['Percentile_75'] = float(series.quantile(0.75))
            
            return stats_dict
        
        # For non-numeric, non-date data
        value_counts = series.value_counts()
        return {
            'Data Type': str(series.dtype),
            'Is Numeric': False,
            'Sample Values': series.head().tolist(),
            'Most Common Values': value_counts.head().to_dict(),
            'Missing Values': int(series.isnull().sum()),
            'Note': 'Column contains non-numeric data. Statistical analysis not applicable.'
        }

    def check_stationarity(self, column: str) -> Dict:
        """
        Perform stationarity tests (ADF and KPSS).
        
        Returns:
            Dict containing test results and suggestions.
        """
        series = self.df[column]
        
        # ADF Test
        adf_result = adfuller(series, autolag='AIC')
        
        # KPSS Test
        kpss_result = kpss(series, regression='c', nlags="auto")
        
        stationarity_results = {
            'ADF': {
                'statistic': adf_result[0],
                'p-value': adf_result[1],
                'is_stationary': adf_result[1] < 0.05
            },
            'KPSS': {
                'statistic': kpss_result[0],
                'p-value': kpss_result[1],
                'is_stationary': kpss_result[1] > 0.05
            }
        }
        
        # Add recommendations
        stationarity_results['recommendations'] = self._get_stationarity_recommendations(
            stationarity_results
        )
        
        return stationarity_results

    def detect_seasonality(self, column: str) -> Dict:
        """
        Detect and analyze seasonality in the time series.
        
        Returns:
            Dict containing seasonality information and decomposition.
        """
        series = self.df[column]
        
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(
            series,
            period=self._estimate_seasonal_period(series),
            extrapolate_trend='freq'
        )
        
        # Calculate strength of seasonality
        seasonal_strength = 1 - (
            np.var(decomposition.resid) / np.var(decomposition.seasonal + decomposition.resid)
        )
        
        return {
            'has_seasonality': seasonal_strength > 0.3,
            'seasonal_strength': seasonal_strength,
            'decomposition': decomposition,
            'suggested_period': self._estimate_seasonal_period(series)
        }

    def detect_outliers(self, column: str, method: str = 'zscore') -> Dict:
        """
        Detect outliers using various methods.
        
        Args:
            column: Target column name
            method: 'zscore', 'iqr', or 'isolation_forest'
            
        Returns:
            Dict containing outlier indices and statistics
        """
        series = self.df[column]
        outliers_dict = {}
        
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(series))
            outliers_dict['indices'] = np.where(z_scores > 3)[0]
            outliers_dict['method'] = 'Z-Score (threshold: 3)'
            
        elif method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            outliers_dict['indices'] = series[
                (series < (Q1 - 1.5 * IQR)) | 
                (series > (Q3 + 1.5 * IQR))
            ].index
            outliers_dict['method'] = 'IQR'
            
        outliers_dict['count'] = len(outliers_dict['indices'])
        outliers_dict['percentage'] = (len(outliers_dict['indices']) / len(series)) * 100
        
        return outliers_dict

    def analyze_missing_values(self, column: str) -> Dict:
        """Analyze missing values and suggest imputation strategies."""
        series = self.df[column]
        
        missing_info = {
            'total_missing': series.isnull().sum(),
            'percentage_missing': (series.isnull().sum() / len(series)) * 100,
            'missing_patterns': self._analyze_missing_patterns(series)
        }
        
        # Add recommendations based on patterns
        missing_info['recommendations'] = self._get_imputation_recommendations(
            missing_info['missing_patterns']
        )
        
        return missing_info

    def generate_feature_recommendations(self, column: str) -> List[Dict]:
        """Generate recommendations for feature engineering."""
        recommendations = []
        
        # Check if rolling statistics would be useful
        if self._has_trend(self.df[column]):
            recommendations.append({
                'type': 'rolling_statistics',
                'description': 'Add rolling mean and standard deviation features',
                'suggested_windows': [7, 14, 30]
            })
        
        # Check if lag features would be useful
        if self._has_autocorrelation(self.df[column]):
            recommendations.append({
                'type': 'lag_features',
                'description': 'Add lagged versions of the target variable',
                'suggested_lags': self._suggest_lag_values(self.df[column])
            })
        
        # Check if difference features would be useful
        stationarity_results = self.check_stationarity(column)
        if not stationarity_results['ADF']['is_stationary']:
            recommendations.append({
                'type': 'difference_features',
                'description': 'Add first and/or seasonal differences',
                'suggested_differences': ['first_difference', 'seasonal_difference']
            })
        
        return recommendations

    def _estimate_seasonal_period(self, series: pd.Series) -> int:
        """Estimate the seasonal period of the time series."""
        if isinstance(series.index, pd.DatetimeIndex):
            # Check common periods
            if len(series) >= 365:
                return 365  # Daily data
            elif len(series) >= 52:
                return 52  # Weekly data
            elif len(series) >= 12:
                return 12  # Monthly data
            elif len(series) >= 4:
                return 4   # Quarterly data
        
        # Fallback to frequency analysis
        from scipy import fftpack
        fft = fftpack.fft(series.values)
        frequencies = fftpack.fftfreq(len(series))
        positive_frequencies = frequencies[frequencies > 0]
        magnitudes = abs(fft)[frequencies > 0]
        peak_frequency = positive_frequencies[magnitudes.argmax()]
        
        if peak_frequency == 0:
            return 1
        return int(round(1/peak_frequency))

    def _has_trend(self, series: pd.Series) -> bool:
        """Check if series has a trend using Mann-Kendall test."""
        from scipy.stats import kendalltau
        tau, p_value = kendalltau(range(len(series)), series)
        return p_value < 0.05

    def _has_autocorrelation(self, series: pd.Series) -> bool:
        """Check for significant autocorrelation."""
        lb_test = acorr_ljungbox(series, lags=[10], return_df=True)
        return lb_test['lb_pvalue'].iloc[0] < 0.05

    def _suggest_lag_values(self, series: pd.Series) -> List[int]:
        """Suggest optimal lag values based on autocorrelation."""
        from statsmodels.tsa.stattools import pacf
        nlags = min(len(series) // 4, 100)  # Maximum number of lags to consider
        pacf_values = pacf(series, nlags=nlags)
        significant_lags = [i for i, v in enumerate(pacf_values) if abs(v) > 2/np.sqrt(len(series))]
        return sorted(significant_lags)[:5]  # Return top 5 significant lags

    def _analyze_missing_patterns(self, series: pd.Series) -> Dict:
        """Analyze patterns in missing values."""
        missing_mask = series.isnull()
        
        # Calculate run lengths of missing values
        runs = pd.Series(missing_mask).groupby(
            (missing_mask != missing_mask.shift()).cumsum()
        ).agg(['size', 'count'])
        
        return {
            'max_consecutive': runs['size'].max() if not runs.empty else 0,
            'avg_gap_length': runs['size'].mean() if not runs.empty else 0,
            'pattern': 'random' if len(runs) > 1 else 'single_block'
        }

    def _get_imputation_recommendations(self, patterns: Dict) -> List[str]:
        """Generate imputation recommendations based on missing patterns."""
        recommendations = []
        
        if patterns['max_consecutive'] <= 3:
            recommendations.append("Linear interpolation suitable for short gaps")
        if patterns['pattern'] == 'random':
            recommendations.append("KNN or MICE imputation recommended for random missing values")
        if patterns['avg_gap_length'] > 5:
            recommendations.append("Consider advanced time series imputation methods")
            
        return recommendations

    def _get_stationarity_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on stationarity test results."""
        recommendations = []
        
        if not results['ADF']['is_stationary']:
            recommendations.append("Consider differencing the series")
            recommendations.append("Try log transformation")
        
        if not results['KPSS']['is_stationary']:
            recommendations.append("Remove trend using detrending or differencing")
            
        return recommendations

    def plot_analysis_results(self, column: str) -> None:
        """Create comprehensive analysis plots using plotly."""
        try:
            # Create a unique identifier using column name and random string
            import uuid
            unique_id = str(uuid.uuid4())[:8]

            # Time Series and Distribution
            fig1 = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Time Series', 'Distribution')
            )

            series = self.df[column]

            # Time Series Plot
            fig1.add_trace(
                go.Scatter(x=self.df.index, y=series, name='Time Series'),
                row=1, col=1
            )

            # Distribution Plot
            fig1.add_trace(
                go.Histogram(x=series, name='Distribution', nbinsx=30),
                row=1, col=2
            )

            st.plotly_chart(fig1, use_container_width=True, key=f'ts_dist_{unique_id}_1')

            # Seasonal and ACF
            fig2 = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Seasonal Pattern', 'Autocorrelation')
            )

            # Seasonal Plot
            try:
                decomp_result = self.detect_seasonality(column)
                if 'decomposition' in decomp_result:
                    decomp = decomp_result['decomposition']
                    fig2.add_trace(
                        go.Scatter(x=self.df.index, y=decomp.seasonal, name='Seasonal'),
                        row=1, col=1
                    )
            except Exception as e:
                fig2.add_annotation(
                    text="Could not compute seasonal decomposition",
                    xref="x domain", yref="y domain",
                    x=0.5, y=0.5, showarrow=False,
                    row=1, col=1
                )

            # ACF Plot
            try:
                from statsmodels.tsa.stattools import acf
                if len(series.dropna()) >= 2:
                    acf_values = acf(series.dropna(), nlags=min(40, len(series) // 2))
                    fig2.add_trace(
                        go.Scatter(x=np.arange(len(acf_values)), y=acf_values, name='ACF'),
                        row=1, col=2
                    )
            except Exception as e:
                fig2.add_annotation(
                    text="Could not compute ACF",
                    xref="x domain", yref="y domain",
                    x=0.5, y=0.5, showarrow=False,
                    row=1, col=2
                )

            st.plotly_chart(fig2, use_container_width=True, key=f'seasonal_acf_{unique_id}_2')

            # Missing Values and Box Plot
            fig3 = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Missing Values Pattern', 'Box Plot')
            )

            # Missing Values Plot
            missing_mask = series.isnull().astype(int)
            fig3.add_trace(
                go.Scatter(x=self.df.index, y=missing_mask, mode='markers',
                        name='Missing'),
                row=1, col=1
            )

            # Box Plot
            fig3.add_trace(
                go.Box(y=series, name='Distribution'),
                row=1, col=2
            )

            st.plotly_chart(fig3, use_container_width=True, key=f'missing_box_{unique_id}_3')

        except Exception as e:
            st.error(f"Error generating plots: {str(e)}")