"""Machine learning model implementations for time series forecasting."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from .base import TimeSeriesModel
import streamlit as st


class RandomForestModel(TimeSeriesModel):
    """Random Forest model implementation."""

    def __init__(self, name="Random Forest"):
        super().__init__(name)
        self.n_estimators = 100
        self.max_depth = None
        self.random_state = 42

    def build(self, n_estimators=100, max_depth=None, random_state=42):
        """Build Random Forest model."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        return self

    def train(self, train_data, val_data=None, **kwargs):
        """Train the model."""
        try:
            X_train, y_train = train_data
            self.model.fit(X_train, y_train)
            return self
        except Exception as e:
            st.error(f"Error training Random Forest model: {str(e)}")
            return None

    def evaluate(self, test_data):
        """Evaluate model performance."""
        try:
            X_test, y_test = test_data
            predictions = self.model.predict(X_test)

            mae = np.mean(np.abs(predictions - y_test))
            rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
            mape = np.mean(np.abs((predictions - y_test) / y_test)) * 100

            return {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape
            }
        except Exception as e:
            st.error(f"Error evaluating Random Forest model: {str(e)}")
            return None

    def predict(self, X):
        """Generate predictions."""
        try:
            return self.model.predict(X)
        except Exception as e:
            st.error(f"Error in Random Forest prediction: {str(e)}")
            return None

    def save(self, path):
        """Save model to file."""
        try:
            import joblib
            joblib.dump(self.model, path)
        except Exception as e:
            st.error(f"Error saving Random Forest model: {str(e)}")

    def load(self, path):
        """Load model from file."""
        try:
            import joblib
            self.model = joblib.load(path)
        except Exception as e:
            st.error(f"Error loading Random Forest model: {str(e)}")


class XGBoostModel(TimeSeriesModel):
    """XGBoost model implementation."""

    def __init__(self, name="XGBoost"):
        super().__init__(name)
        self.n_estimators = 100
        self.max_depth = 6
        self.learning_rate = 0.1
        self.random_state = 42

    def build(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
        """Build XGBoost model."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state
        )
        return self

    def train(self, train_data, val_data=None, **kwargs):
        """Train the model."""
        try:
            X_train, y_train = train_data
            eval_set = [(X_train, y_train)]

            if val_data is not None:
                X_val, y_val = val_data
                eval_set.append((X_val, y_val))

            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=10,
                verbose=False
            )
            return self
        except Exception as e:
            st.error(f"Error training XGBoost model: {str(e)}")
            return None

    def evaluate(self, test_data):
        """Evaluate model performance."""
        try:
            X_test, y_test = test_data
            predictions = self.model.predict(X_test)

            mae = np.mean(np.abs(predictions - y_test))
            rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
            mape = np.mean(np.abs((predictions - y_test) / y_test)) * 100

            return {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape
            }
        except Exception as e:
            st.error(f"Error evaluating XGBoost model: {str(e)}")
            return None

    def predict(self, X):
        """Generate predictions."""
        try:
            return self.model.predict(X)
        except Exception as e:
            st.error(f"Error in XGBoost prediction: {str(e)}")
            return None

    def save(self, path):
        """Save model to file."""
        try:
            self.model.save_model(path)
        except Exception as e:
            st.error(f"Error saving XGBoost model: {str(e)}")

    def load(self, path):
        """Load model from file."""
        try:
            self.model.load_model(path)
        except Exception as e:
            st.error(f"Error loading XGBoost model: {str(e)}")


def create_ml_model(model_type, **kwargs):
    """Factory function to create ML models."""
    if model_type.upper() == 'RANDOM FOREST':
        return RandomForestModel(**kwargs)
    elif model_type.upper() == 'XGBOOST':
        return XGBoostModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")