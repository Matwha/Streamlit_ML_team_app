def generate_prediction_intervals(
    df,
    target_variable,
    train_size=0.8,
    lags=12,
    forecast_periods=12,
    model=RandomForestRegressor(),
    method="conformal",  # "conformal" "bootstrapping" or "quantile"
    prediction_type="one-step",  # "one-step" or "multi-step"
    alpha=0.05,
    n_bootstraps=100,
    conformal_cv=BlockBootstrap(n_resamplings=100, length=24, overlapping=True),
    lower_q=0.025,
    upper_q=0.975
):
    """
    Generate prediction intervals using bootstrapping or conformal prediction.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the time series data.
    target_variable : str
        Name of the target column.
    train_size : float
        Proportion of data used for training.
    lags : int
        Number of lagged features.
    forecast_periods : int
        Number of periods to forecast (for multi-step).
    model : scikit-learn compatible model
        Prediction model.
    method : str
        "bootstrapping" or "conformal".
    prediction_type : str
        "one-step" or "multi-step".
    alpha : float
        Significance level for confidence intervals (1-alpha confidence interval).
    n_bootstraps : int
        Number of bootstrap iterations (for bootstrapping).
    conformal_cv : object
        Conformal cross-validation object.
    lower_q : float
        Lower quantile (e.g., 0.025 for 95% interval).
    upper_q : float
        Upper quantile (e.g., 0.975 for 95% interval).

    Returns:
    --------
    results : pd.DataFrame
        DataFrame containing actual, predicted, and confidence intervals.
    """

    # Generate lagged features
    def create_lagged_features(df, target_variable, lags):
        for lag in range(1, lags + 1):
            df[f'lag_{lag}'] = df[target_variable].shift(lag)
        df.dropna(inplace=True)
        features = [f'lag_{lag}' for lag in range(1, lags + 1)]
        X = df[features]
        y = df[target_variable]
        return X, y

    X, y = create_lagged_features(df, target_variable, lags)

    # Train/test split
    train_size_idx = int(len(X) * train_size)
    X_train, X_test = X[:train_size_idx], X[train_size_idx:]
    y_train, y_test = y[:train_size_idx], y[train_size_idx:]

    results = None

    if method == "conformal":
        # Conformal Prediction
        mapie = MapieTimeSeriesRegressor(model, method="enbpi", cv=conformal_cv, n_jobs=-1)
        mapie.fit(X_train, y_train)

        if prediction_type == "one-step":
            y_pred, y_pis = mapie.predict(X_test, alpha=alpha)
            results = pd.DataFrame({
                "actual": y_test,
                "prediction": y_pred,
                "lower_bound": y_pis[:, 0, 0],
                "upper_bound": y_pis[:, 1, 0]
            }, index=y_test.index)
        elif prediction_type == "multi-step":
            last_known = y_train.iloc[-lags:].values
            predictions, lower_bounds, upper_bounds = [], [], []
            for _ in range(forecast_periods):
                X_step = pd.DataFrame([{f'lag_{i+1}': last_known[-(i+1)] for i in range(lags)}])
                X_step = X_step[X_train.columns]
                y_pred, y_pis = mapie.predict(X_step, alpha=alpha)
                predictions.append(y_pred[0])
                lower_bounds.append(y_pis[0, 0, 0])
                upper_bounds.append(y_pis[0, 1, 0])
                last_known = np.append(last_known[1:], y_pred[0])

            future_index = pd.date_range(
                start=df.index[-1],
                periods=forecast_periods + 1,
                freq=pd.infer_freq(df.index)
            )[1:]
            results = pd.DataFrame({
                "actual": y_test,
                "prediction": predictions,
                "lower_bound": lower_bounds,
                "upper_bound": upper_bounds
            }, index=future_index)
            
    elif method == "bootstrapping":
        # Bootstrapping
        model.fit(X_train, y_train)

        if prediction_type == "one-step":
            # One-step predictions
            y_pred = model.predict(X_test)
            bootstrap_preds = []
            for _ in range(n_bootstraps):
                X_resampled, y_resampled = resample(X_train, y_train)
                model.fit(X_resampled, y_resampled)
                bootstrap_preds.append(model.predict(X_test))

            bootstrap_preds = np.array(bootstrap_preds)
            lower_bounds = np.percentile(bootstrap_preds, 100 * (alpha / 2), axis=0)
            upper_bounds = np.percentile(bootstrap_preds, 100 * (1 - alpha / 2), axis=0)

            results = pd.DataFrame({
                "actual": y_test,
                "prediction": y_pred,
                "lower_bound": lower_bounds,
                "upper_bound": upper_bounds
            }, index=y_test.index)
            
        elif prediction_type == "multi-step":
            # Multi-step predictions
            last_known = X.iloc[-1].values
            predictions, lower_bounds, upper_bounds = [], [], []
            for _ in range(forecast_periods):
                future_pred = model.predict(last_known.reshape(1, -1))[0]
                predictions.append(future_pred)

                bootstrap_preds = []
                for _ in range(n_bootstraps):
                    X_resampled, y_resampled = resample(X_train, y_train)
                    model.fit(X_resampled, y_resampled)
                    bootstrap_preds.append(model.predict(last_known.reshape(1, -1))[0])

                lower_bounds.append(np.percentile(bootstrap_preds, 100 * (alpha / 2)))
                upper_bounds.append(np.percentile(bootstrap_preds, 100 * (1 - alpha / 2)))

                last_known = np.roll(last_known, -1)
                last_known[-1] = future_pred

            future_index = pd.date_range(
                start=df.index[-1],
                periods=forecast_periods + 1,
                freq=pd.infer_freq(df.index)
            )[1:]
            results = pd.DataFrame({
                "actual": y_test,
                "prediction": predictions,
                "lower_bound": lower_bounds,
                "upper_bound": upper_bounds
            }, index=future_index)
        pass
            
    elif method == "quantile":
        # Quantile Regression
        lower_model = QuantileRegressor(quantile=lower_q, alpha=0, solver='highs')
        upper_model = QuantileRegressor(quantile=upper_q, alpha=0, solver='highs')
        median_model = QuantileRegressor(quantile=0.5, alpha=0, solver='highs')

        # Fit models
        lower_model.fit(X_train, y_train)
        upper_model.fit(X_train, y_train)
        median_model.fit(X_train, y_train)

        if prediction_type == "one-step":
            # One-step predictions
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
            
        elif prediction_type == "multi-step":
            # Multi-step predictions
            last_known = X.iloc[-1].values
            predictions, lower_bounds, upper_bounds = [], [], []

            for _ in range(forecast_periods):
                # Predict quantiles for the next step
                lower_pred = lower_model.predict(last_known.reshape(1, -1))[0]
                upper_pred = upper_model.predict(last_known.reshape(1, -1))[0]
                median_pred = median_model.predict(last_known.reshape(1, -1))[0]

                predictions.append(median_pred)
                lower_bounds.append(lower_pred)
                upper_bounds.append(upper_pred)

                # Update lagged features for the next step
                last_known = np.roll(last_known, -1)
                last_known[-1] = median_pred

            # Create future index
            future_index = pd.date_range(
                start=series.index[-1],
                periods=forecast_periods + 1,
                freq=pd.infer_freq(series.index)
            )[1:]

            # Create results DataFrame
            results = pd.DataFrame({
                "actual": y_test,
                'prediction': predictions,
                'lower_bound': lower_bounds,
                'upper_bound': upper_bounds
            }, index=future_index)
        pass

    return results

with st.sidebar:
    st.subheader("Confidence Interval Settings")
    method = st.selectbox("Confidence Interval Method", ["bootstrapping", "quantile", "conformal"])
    prediction_type = st.radio("Prediction Type", ["one-step", "multi-step"])
    alpha = st.slider("Significance Level (alpha)", min_value=0.01, max_value=0.2, step=0.01, value=0.05)
    if method == "bootstrapping":
        n_bootstraps = st.slider("Number of Bootstraps", min_value=10, max_value=200, step=10, value=100)
    elif method == "quantile":
        lower_q = st.slider("Lower Quantile", min_value=0.01, max_value=0.5, step=0.01, value=0.025)
        upper_q = st.slider("Upper Quantile", min_value=0.5, max_value=0.99, step=0.01, value=0.975)

if st.button("Run Forecast Intervals"):
    results = generate_prediction_intervals(
        df=df,
        target_variable=target_variable,
        train_size=train_size,
        lags=lags,
        forecast_periods=forecast_periods,
        method=method,
        prediction_type=prediction_type,
        alpha=alpha,
        n_bootstraps=n_bootstraps if method == "bootstrapping" else None,
        lower_q=lower_q if method == "quantile" else None,
        upper_q=upper_q if method == "quantile" else None
    )

def plot_interactive_forecast(
    results,
    title="Forecast with Prediction Intervals",
    xlabel="Date",
    ylabel="Total Sales"
):
    fig = go.Figure()

    # Add predictions
    fig.add_trace(go.Scatter(
        x=results.index, 
        y=results['prediction'], 
        mode='lines', 
        name='Prediction',
        line=dict(color='blue')
    ))

    # Add actual values
    if 'actual' in results.columns:
        fig.add_trace(go.Scatter(
            x=results.index, 
            y=results['actual'], 
            mode='lines', 
            name='Actual',
            line=dict(color='black', dash='dot')
        ))

    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=results.index.tolist() + results.index[::-1].tolist(),
        y=results['upper_bound'].tolist() + results['lower_bound'][::-1].tolist(),
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Prediction Interval'
    ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template="plotly_white"
    )

    return fig

if st.button("Plot Forecast"):
    fig = plot_interactive_forecast(results)
    st.plotly_chart(fig, use_container_width=True)