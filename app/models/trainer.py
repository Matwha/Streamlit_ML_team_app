"""Model training and evaluation utilities."""

import tensorflow as tf
from tensorflow import keras
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


class ModelTrainer:
    """Enhanced class for handling model training and evaluation."""

    def __init__(self, model, target_column):
        self.model = model
        self.target_column = target_column

    def prepare_training_data(self, df: pd.DataFrame, model_name: str, sequence_length: int = 10,
                              batch_size: int = 32) -> Dict:
        """Prepare data for training based on model type."""
        try:
            if model_name in ['Simple RNN', 'LSTM', 'Stacked LSTM+RNN']:
                return self._prepare_deep_learning_data(
                    df,
                    sequence_length=sequence_length,
                    batch_size=batch_size
                )
            else:
                # For other model types, return the raw data
                train_size = int(len(df) * 0.8)
                return {
                    'train': df[:train_size],
                    'test': df[train_size:],
                    'test_index': df.index[train_size:]
                }
        except Exception as e:
            st.error(f"Error preparing training data: {str(e)}")
            return None

    def _prepare_deep_learning_data(self, df: pd.DataFrame, sequence_length: int = 12, batch_size: int = 32) -> Dict:
        """Prepare sequences for deep learning models."""
        try:
            # Get the target series
            series = df[self.target_column].values

            # Create sequences
            sequences = []
            targets = []

            # Create sequence/target pairs
            for i in range(len(series) - sequence_length):
                sequence = series[i:(i + sequence_length)]
                target = series[i + sequence_length]
                sequences.append(sequence)
                targets.append(target)

            # Convert to numpy arrays
            X = np.array(sequences)
            y = np.array(targets)

            # Reshape for RNN input [samples, timesteps, features]
            X = X.reshape((X.shape[0], X.shape[1], 1))

            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # Create tensorflow datasets with proper batching
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
            test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

            # Store indices for later use
            test_index = df.index[train_size + sequence_length:]

            # Return all necessary components
            return {
                'train': train_dataset,
                'test': test_dataset,
                'test_target': pd.Series(y_test, index=test_index),
                'test_index': test_index,
                'X_test': X_test,
                'y_test': y_test,
                'sequence_length': sequence_length
            }

        except Exception as e:
            st.error(f"Error preparing deep learning data: {str(e)}")
            st.write("Debug info:")
            st.write(f"Sequence length: {sequence_length}")
            st.write(f"Initial data shape: {series.shape}")
            st.write(f"Sequences shape before reshape: {np.array(sequences).shape}")
            return None

    def train_and_evaluate(self, df: pd.DataFrame, model_name: str,
                          model_params: Dict) -> Tuple[Optional[object],
                                                     Optional[pd.DataFrame],
                                                     Optional[Dict]]:
        """Train model and evaluate results."""
        try:
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Prepare data
            status_text.text("Preparing data...")
            progress_bar.progress(0.2)

            data_splits = self.prepare_training_data(
                df,
                model_name,
                sequence_length=model_params['build_params'].get('sequence_length', 12),
                batch_size=model_params['train_params'].get('batch_size', 32)
            )

            if data_splits is None:
                st.error("Failed to prepare data splits")
                return None, None, None

            # Train model
            status_text.text("Training model...")
            progress_bar.progress(0.4)

            # Update train_params dictionary instead of passing parameters directly
            train_params = {
                'epochs': model_params['train_params'].get('epochs', 100),
                'batch_size': model_params['train_params'].get('batch_size', 32),
                'callbacks': self.create_callbacks(model_name),
                'n_iterations': model_params['train_params'].get('n_iterations', 100),
                'confidence_level': model_params['train_params'].get('confidence_level', 0.95),
                'interval_method': model_params['train_params'].get('interval_method', 'bootstrap'),
                'patience': model_params['train_params'].get('patience', 5),
                'learning_rate': model_params['train_params'].get('learning_rate', 0.001)
            }

            # Set model's train_params
            self.model.train_params = train_params

            # Call train method with data only
            history = self.model.train(
                data_splits['train'],
                val_data=data_splits.get('test')
            )

            progress_bar.progress(0.6)

            # Generate predictions
            status_text.text("Generating predictions...")
            if model_name in ['Simple RNN', 'LSTM', 'Stacked LSTM+RNN']:
                predictions = self.model.predict(data_splits['X_test'])
            else:
                predictions = self.model.predict(data_splits['test'])

            # Calculate metrics
            metrics = self.model.evaluate(data_splits['test'])

            # Create results DataFrame
            results_df = pd.DataFrame({
                'Actual': data_splits['test_target'],
                'Predicted': predictions
            }, index=data_splits['test_index'])

            # Add confidence intervals if requested
            if st.session_state.get('display_settings', {}).get('show_ci', False):
                ci_level = st.session_state['display_settings'].get('ci_level', 0.95)
                mean_pred, lower, upper = self._calculate_prediction_intervals(
                    data_splits['test'],
                    predictions,
                    ci_level
                )
                results_df['Lower Bound'] = lower
                results_df['Upper Bound'] = upper

            progress_bar.progress(1.0)
            status_text.text("Analysis complete!")

            return self.model, results_df, metrics

        except Exception as e:
            st.error(f"Error in model training and evaluation: {str(e)}")
            with st.expander("🔍 Debug: Error Details", expanded=True):
                st.error(f"Error Type: {type(e).__name__}")
                st.error(f"Error Message: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
            return None, None, None

    def _calculate_prediction_intervals(self, test_data, predictions, confidence=0.95):
        """Calculate prediction intervals using bootstrap method."""
        try:
            if isinstance(test_data, tf.data.Dataset):
                # For deep learning models, extract the raw data
                test_data = np.vstack([x.numpy() for x, _ in test_data])

            residuals = predictions - test_data
            std_dev = np.std(residuals)
            z_score = stats.norm.ppf(1 - (1 - confidence) / 2)

            lower = predictions - z_score * std_dev
            upper = predictions + z_score * std_dev

            return predictions, lower, upper

        except Exception as e:
            st.error(f"Error calculating prediction intervals: {str(e)}")
            return predictions, predictions, predictions

    def create_callbacks(self, model_name):
        """Create training callbacks."""
        class CustomCallback(keras.callbacks.Callback):
            def __init__(self, progress_bar, status_text, metrics_container):
                super().__init__()
                self.progress_bar = progress_bar
                self.status_text = status_text
                self.metrics_container = metrics_container

            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / self.params['epochs']
                self.progress_bar.progress(progress)
                self.status_text.text(f"Training epoch {epoch + 1}/{self.params['epochs']}")
                if logs:
                    metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
                    self.metrics_container.text(f"Current metrics: {metrics_str}")

        # Create Streamlit progress containers
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_container = st.empty()

        return [
            CustomCallback(progress_bar, status_text, metrics_container),
            keras.callbacks.ModelCheckpoint(
                f"{self.target_column}_{model_name.lower().replace(' ', '_')}.keras",
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

    def plot_training_history(self, history):
        """Plot training history."""
        if hasattr(history, 'history'):
            fig, ax = plt.subplots()
            ax.plot(history.history['loss'], label='Training Loss')
            if 'val_loss' in history.history:
                ax.plot(history.history['val_loss'], label='Validation Loss')
            ax.set_title(f'{self.model.name} Training History')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)