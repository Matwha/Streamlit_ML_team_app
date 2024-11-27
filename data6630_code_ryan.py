import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.svm import SVR

from mapie.regression import MapieTimeSeriesRegressor
from mapie.subsample import BlockBootstrap

import warnings
warnings.filterwarnings('ignore')

# !unzip /content/truck_data.zip

# Preprocess and EDA
truck_df = pd.read_csv('/content/truck_sales.csv')

truck_df.head()

truck_df['date'] = pd.to_datetime(truck_df['Month-Year'], format='%y-%b')
truck_df.drop('Month-Year', axis=1, inplace=True)

truck_df = truck_df.set_index('date')
truck_df.index = pd.to_datetime(truck_df.index)

truck_df.info()

truck_df.head()

truck_df.rename(columns={'Number_Trucks_Sold': 'num_sold'}, inplace=True)

df = truck_df.copy()

df.head()

df.shape

df['Lag_1'] = df['num_sold'].shift(1)
df['Lag_2'] = df['num_sold'].shift(2)

df.dropna(inplace=True)
df.head()

df['num_sold'].plot(figsize=(12, 8))
plt.title('Monthly Number of Trucks Sold')
plt.xlabel('Date')
plt.ylabel('Number of Trucks Sold')
plt.legend()
plt.show()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(df['Lag_1'], df['Lag_2'], c=df['num_sold'])
plt.title('Lag 1 vs Lag 2')
plt.xlabel('Lag 1')
plt.ylabel('Lag 2')
plt.colorbar(label='Number of Trucks Sold')
plt.show()

df['Lag_1'].sort_values().plot(kind='hist', bins=20)
plt.show()

# 

def ml_ts_forecaster(data, target, lags, fh, model, model_name):
    series = data[target].dropna().to_numpy()

    Tx = lags
    Ty = 1  # Forecasting Ty step ahead

    # Splitting the data
    test_period = fh
    train_period = len(data) - test_period
    train = data.iloc[:train_period]
    test = data.iloc[train_period:]

    # boolean series for train and test
    train_indicator = (data.index <= train.index[-1])
    test_indicator =  (data.index > train.index[-1])
    train_indicator[:Tx] = False  # the first Tx values are not predictable.

    # making supervised data
    X = np.array([series[t:t+Tx] for t in range(len(series) - Tx-Ty+1)])
    Y = np.array([series[t+Tx+Ty-1] for t in range(len(series) - Tx-Ty+1)])
    Xtrain, Ytrain = X[:-test_period], Y[:-test_period]
    Xtest, Ytest = X[-test_period:], Y[-test_period:]

    # training the model
    model.fit(Xtrain, Ytrain)

    # one-step ahead forecast
    data.loc[train_indicator, f'{model_name}_1step_train_forecast'] = model.predict(Xtrain)
    data.loc[test_indicator, f'{model_name}_1step_test_forecast'] = model.predict(Xtest)

    # multi-step ahead forecast
    multistep_predictions = []
    input_X = Xtest[0]
    while len(multistep_predictions) < test_period:
        prediction = model.predict(input_X.reshape(1, -1))[0]
        multistep_predictions.append(prediction)
        input_X = np.roll(input_X, -1)
        input_X[-1] = prediction

    data.loc[test_indicator, f'{model_name}_multistep_test_forecast'] = multistep_predictions

    # Error metrics
    # MAPE
    mape_1step = mean_absolute_percentage_error(data.loc[test_indicator, target], data.loc[test_indicator, f'{model_name}_1step_test_forecast'])
    mape_multiple_step = mean_absolute_percentage_error(data.loc[test_indicator, target], data.loc[test_indicator, f'{model_name}_multistep_test_forecast'])

    # RMSE
    rmse_1step = mean_squared_error(data.loc[test_indicator, target], data.loc[test_indicator, f'{model_name}_1step_test_forecast'], squared=False)
    rmse_multistep = mean_squared_error(data.loc[test_indicator, target], data.loc[test_indicator, f'{model_name}_multistep_test_forecast'], squared=False)

    # MAE
    mae_1step = mean_absolute_error(data.loc[test_indicator, target], data.loc[test_indicator, f'{model_name}_1step_test_forecast'])
    mae_multistep = mean_absolute_error(data.loc[test_indicator, target], data.loc[test_indicator, f'{model_name}_multistep_test_forecast'])

    # Create a DataFrame with error metrics
    error_metrics = pd.DataFrame({
        'Model': [model_name, model_name],
        'Step_Type': ['1-step', 'Multi-step'],
        'MAPE': [mape_1step, mape_multiple_step],
        'RMSE': [rmse_1step, rmse_multistep],
        'MAE': [mae_1step, mae_multistep]
    })

    # plot 1-step and multi-step forecast
    data[[target, f'{model_name}_1step_test_forecast', f'{model_name}_multistep_test_forecast']].plot(figsize=(15, 5))
    plt.title('1-step and multi-step ahead forecast')
    plt.show()

    return data, error_metrics

from sklearn.linear_model import QuantileRegressor

def create_1step_ahead_quantile_prediction_intervals(
    data,
    target_col,
    n_lags=12,
    train_size=0.90,
    random_state=101,
    lower_q=0.025,
    upper_q=0.975
):
    """
    Create one-step ahead prediction intervals using quantile regression

    Parameters:
    -----------
    data : pd.DataFrame or pd.Series
        Time series data
    target_col : str
        Name of target column (only needed if data is DataFrame)
    n_lags : int
        Number of lags to create
    train_size : float
        Proportion of data to use for training (0 to 1)
    random_state : int
        Random seed for reproducibility
    lower_q : float
        Lower quantile (e.g., 0.025 for 95% interval)
    upper_q : float
        Upper quantile (e.g., 0.975 for 95% interval)

    Returns:
    --------
    pd.DataFrame
        DataFrame containing actual values, predictions, and prediction intervals
    """
    # Set random seed
    np.random.seed(random_state)

    # Convert to series if DataFrame
    if isinstance(data, pd.DataFrame):
        series = data[target_col]
    else:
        series = data

    # Create features
    df = pd.DataFrame()
    for i in range(1, n_lags + 1):
        df[f'lag_{i}'] = series.shift(i)
    df['target'] = series
    df = df.dropna()

    # Split into X and y
    y = df['target']
    X = df.drop('target', axis=1)

    # Split into train/test
    train_size = int(len(X) * train_size)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Fit quantile regressors
    lower_model = QuantileRegressor(quantile=lower_q, alpha=0, solver='highs')
    upper_model = QuantileRegressor(quantile=upper_q, alpha=0, solver='highs')
    median_model = QuantileRegressor(quantile=0.5, alpha=0, solver='highs')

    # Fit models
    lower_model.fit(X_train, y_train)
    upper_model.fit(X_train, y_train)
    median_model.fit(X_train, y_train)

    # Make predictions
    lower_pred = lower_model.predict(X_test)
    upper_pred = upper_model.predict(X_test)
    median_pred = median_model.predict(X_test)

    # Create results DataFrame
    results = pd.DataFrame({
        'actual': y_test,
        'prediction': median_pred,
        'lower_bound': lower_pred,
        'upper_bound': upper_pred
    }, index=y_test.index)

    # Calculate metrics
    mape = np.mean(np.abs((results['actual'] - results['prediction']) / results['actual'])) * 100
    rmse = np.sqrt(np.mean((results['actual'] - results['prediction'])**2))

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot training data
    plt.plot(y_train.index, y_train, label='Training Data', alpha=0.5)

    # Plot test data and predictions
    plt.plot(y_test.index, y_test, label='Actual', color='black', linewidth=2, alpha=0.3)
    plt.plot(y_test.index, median_pred, label='Predictions', color='blue', linewidth=2)

    # Plot prediction intervals
    plt.fill_between(y_test.index, lower_pred, upper_pred,
                     alpha=0.2, color='blue',
                     label=f'{lower_q*100}% - {upper_q*100}% Prediction Intervals')

    plt.title('One-Step Ahead Forecast with Quantile Regression Prediction Intervals')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"MAPE: {mape:.2f}%")
    print(f"RMSE: {rmse:.2f}")

    # print("\nForecast Summary:")
    # print(results.head())

    return results

# Create and fit the model
def create_onestep_ahead_conformal_prediction_intervals(
    data,
    target_col='passengers',
    model = RandomForestRegressor(),
    n_lags=12,
    train_size=0.90,
    method='enbpi',
    random_state=42
):
    """
    Create one-step ahead prediction intervals using MAPIE

    Parameters:
    -----------
    data : pd.DataFrame
        Time series data
    target_col : str
        Name of target column
    n_lags : int
        Number of lags to create
    train_size : float
        Proportion of data to use for training
    method : str
        MAPIE method ('enbpi' or 'aci')
    random_state : int
        Random seed

    Returns:
    --------
    pd.DataFrame
        DataFrame containing actual values, predictions, and prediction intervals
    """
    # Prepare data
    series = data[target_col]

    # Create features
    df = pd.DataFrame()
    for i in range(1, n_lags + 1):
        df[f'lag_{i}'] = series.shift(i)
    df['target'] = series
    df = df.dropna()

    # Split into X and y
    y = df['target']
    X = df.drop('target', axis=1)

    # Split into train/test
    train_size = int(len(X) * train_size)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Create and fit model

    cv_mapiets = BlockBootstrap(n_resamplings=100, length=24, overlapping=True, random_state=59)
    mapie = MapieTimeSeriesRegressor(
        model,
        method=method,
        cv=cv_mapiets,  # try None and compare
        n_jobs=-1
    )

    # Fit and predict
    mapie.fit(X_train, y_train)
    y_pred, y_pis = mapie.predict(X_test, alpha=0.05)  # 95% prediction interval

    # Create results DataFrame
    results = pd.DataFrame({
        'actual': y_test,
        'prediction': y_pred,
        'lower_bound': y_pis[:, 0, 0],
        'upper_bound': y_pis[:, 1, 0]
    }, index=y_test.index)

    # Calculate metrics
    mape = np.mean(np.abs((results['actual'] - results['prediction']) / results['actual'])) * 100
    rmse = np.sqrt(np.mean((results['actual'] - results['prediction'])**2))

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot training data
    plt.plot(y_train.index, y_train, label='Training Data', alpha=0.5)

    # Plot test data and predictions
    plt.plot(y_test.index, y_test, label='Actual', color='black', linewidth=2, alpha=0.3)
    plt.plot(y_test.index, y_pred, label='Predictions', color='blue', linewidth=2)

    # Plot prediction intervals
    plt.fill_between(y_test.index,
                     y_pis[:, 0, 0],
                     y_pis[:, 1, 0],
                     alpha=0.2, color='blue',
                     label='95% Prediction Intervals')

    plt.title(f'One-Step Ahead Forecast with Conformal Prediction Intervals ({method.upper()})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\nSummary Statistics:")
    print(f"MAPE: {mape:.2f}%")
    print(f"RMSE: {rmse:.2f}")
    print("\nFirst few predictions:")
    print(results.head())

    return results

from mapie.regression import MapieTimeSeriesRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_features(data, n_lags=12):
    """
    Create lag features from time series data
    """
    df = pd.DataFrame()

    # Create lag features
    for i in range(1, n_lags + 1):
        df[f'lag_{i}'] = data.shift(i)

    # Add target
    df['target'] = data

    # Remove rows with NaN values
    df = df.dropna()

    # Split into X and y
    y = df['target']
    X = df.drop('target', axis=1)

    return X, y

def create_multistep_ahead_conformal_prediction_intervals(
    data,
    target_col='passengers',
    model = RandomForestRegressor(),
    n_lags=12,
    train_size=0.90,
    fh=12,
    method='aci',
    random_state=42
):
    """
    Create multi-step ahead prediction intervals using MAPIE
    """
    # Prepare data
    series = data[target_col]
    X, y = create_features(series, n_lags=n_lags)

    # Split into train/test
    train_size = int(len(X) * train_size)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    cv_mapiets = BlockBootstrap(n_resamplings=100, length=24, overlapping=True, random_state=59)
    mapie = MapieTimeSeriesRegressor(
        model,
        method=method,
        cv=cv_mapiets,  # try None and compare
        n_jobs=-1
    )

    # Fit model
    mapie.fit(X_train, y_train)

    # Prepare for recursive forecasting
    last_known = y_train.iloc[-n_lags:].values
    X_future = pd.DataFrame()

    predictions = []
    lower_bounds = []
    upper_bounds = []

    # Make recursive predictions
    for step in range(fh):
        # Prepare features
        X_step = pd.DataFrame([{f'lag_{i+1}': last_known[-(i+1)]
                              for i in range(n_lags)}])

        # Ensure columns are in the same order as training data
        X_step = X_step[X_train.columns]

        # Make prediction with intervals
        y_pred, y_pis = mapie.predict(X_step, alpha=0.05)

        # Store predictions
        predictions.append(y_pred[0])
        lower_bounds.append(y_pis[0, 0, 0])
        upper_bounds.append(y_pis[0, 1, 0])

        # Update last known values
        last_known = np.append(last_known[1:], y_pred[0])

    # Create future dates
    if isinstance(y_train.index, pd.PeriodIndex):
        future_dates = pd.period_range(
            start=y_train.index[-1] + 1,
            periods=fh,
            freq=y_train.index.freq
        )
        plot_dates = future_dates.to_timestamp()
        y_train_plot = y_train.copy()
        y_train_plot.index = y_train.index.to_timestamp()
    else:
        freq = pd.infer_freq(y_train.index)
        future_dates = pd.date_range(
            start=y_train.index[-1],
            periods=fh + 1,
            freq=freq
        )[1:]
        plot_dates = future_dates
        y_train_plot = y_train

    # Create results DataFrame
    results = pd.DataFrame({
        'actual': y_test,
        'prediction': predictions,
        'lower_bound': lower_bounds,
        'upper_bound': upper_bounds
    }, index=future_dates)

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot all training data
    plt.plot(y_train_plot.index, y_train_plot, label='Historical Data', alpha=0.5)

    # Plot the actual values
    plt.plot(y_test.index, y_test, label='Actual Data', alpha=0.7)

    # Plot predictions and intervals
    plt.plot(plot_dates, predictions,
             label='Predictions', color='blue', linewidth=2)
    plt.fill_between(plot_dates, lower_bounds, upper_bounds,
                     alpha=0.2, color='blue',
                     label='95% Prediction Intervals')

    plt.title(f'Multi-step Ahead Forecast with Conformal Prediction Intervals ({method.upper()})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\nForecast Summary:")
    print(results.head())

    return results

"""#XGBoost"""

df = truck_df.copy()
xgboost_data, xgboost_error_metrics = ml_ts_forecaster(data=df, target='num_sold', lags=12, fh=12, model=XGBRegressor(n_estimators=100), model_name='XGBoost')
print(xgboost_error_metrics)

"""## 1-step - Conformal Prediction Intervals"""

xgboost_results_1step_cp = create_onestep_ahead_conformal_prediction_intervals(
    data=df,
    target_col='num_sold',
    model=XGBRegressor(n_estimators=100),
    n_lags=12,
    train_size=0.90,
    method='enbpi'
)

"""## Multi-step - Conformal Prediction Intervals"""

xgboost_results_mstep_cp = create_multistep_ahead_conformal_prediction_intervals(
    data=df,
    target_col='num_sold',
    model=XGBRegressor(n_estimators=100),
    n_lags=12,
    train_size=0.90,
    fh=12,
    method='aci'
)

"""# Random Forest"""

df = truck_df.copy()
rf_data, rf_error_metrics = ml_ts_forecaster(data=df, target='num_sold', lags=12, fh=12, model = RandomForestRegressor(bootstrap=False), model_name='Random Forest')
print(rf_error_metrics)

"""## 1-step - Quantile Regression Intervals"""

rf_results_1step_QR = create_1step_ahead_quantile_prediction_intervals(
    data=df,
    target_col='num_sold',
    n_lags=12,
    train_size=0.90,
    lower_q=0.025,
    upper_q=0.975
)

rf_results_1step_QR.head()

"""## Multi-step - Conformal Prediction Intervals"""

rf_results_mstep_cp = create_multistep_ahead_conformal_prediction_intervals(
    data=df,
    target_col='num_sold',
    model = RandomForestRegressor(bootstrap=False),
    n_lags=12,
    train_size=0.90,
    fh=12,
    method='aci'
)

"""# Log Random Forest"""

df = truck_df.copy()
df['log_num_sold'] = np.log(df['num_sold'])
log_rf_data, log_rf_error_metrics = ml_ts_forecaster(data=df, target='num_sold', lags=12, fh=12, model = RandomForestRegressor(bootstrap=False), model_name='Log Random Forest')
print(log_rf_error_metrics)

"""## 1-step - Quantile Regression Intervals"""

log_rf_results_1step_QR = create_1step_ahead_quantile_prediction_intervals(
    data=df,
    target_col='num_sold',
    n_lags=12,
    train_size=0.90,
    lower_q=0.025,
    upper_q=0.975
)

log_rf_results_1step_QR.head()

"""## Multi-step - Conformal Prediction Intervals"""

_log_rf_results_mstep_cp = create_multistep_ahead_conformal_prediction_intervals(
    data=df,
    target_col='num_sold',
    model = RandomForestRegressor(bootstrap=False),
    n_lags=12,
    train_size=0.90,
    fh=12,
    method='aci'
)

"""# SVR"""

df = truck_df.copy()
svr_data, svr_error_metrics = ml_ts_forecaster(data=df, target='num_sold', lags=12, fh=12, model = SVR(kernel='rbf', C=100), model_name='SVR')
print(svr_error_metrics)

"""## 1-step - Quantile Regression Intervals"""

svr_results_1step_QR = create_1step_ahead_quantile_prediction_intervals(
    data=df,
    target_col='num_sold',
    n_lags=12,
    train_size=0.90,
    lower_q=0.025,
    upper_q=0.975
)
svr_results_1step_QR.head()

"""## Multi-step - Conformal Prediction Intervals"""

svr_results_mstep_cp = create_multistep_ahead_conformal_prediction_intervals(
    data=df,
    target_col='num_sold',
    model = SVR(kernel='rbf', C=100),
    n_lags=12,
    train_size=0.90,
    fh=12,
    method='aci'
)

combined_metrics = pd.concat([xgboost_error_metrics, rf_error_metrics, log_rf_error_metrics, svr_error_metrics], ignore_index=True)
combined_metrics

def highlight_min_max(s):
    is_min = s == s.min()
    is_max = s == s.max()
    # Create an empty list to store styles
    styles = []
    # Iterate over the boolean Series
    for val_min, val_max in zip(is_min, is_max):
        if val_min:  # Check if current value is minimum
            style = 'background-color: green'
        elif val_max:  # Check if current value is maximum
            style = 'background-color: red'
        else:
            style = ''  # No style if not min or max
        styles.append(style)  # Add the style to the list
    return styles  # Return the list of styles


styled_combined_metrics = combined_metrics.style.apply(highlight_min_max, subset=['MAPE', 'RMSE', 'MAE'])
styled_combined_metrics