"""Model training and evaluation utilities."""

import tensorflow as tf
from tensorflow import keras
import streamlit as st
import matplotlib.pyplot as plt


class ModelTrainer:
    """Class for handling model training and evaluation."""

    def __init__(self, model, target_column):
        self.model = model
        self.target_column = target_column

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

    def train_and_evaluate(self, train_dataset, val_dataset, test_dataset, epochs):
        """Train and evaluate the model."""
        try:
            st.write("Debug: Training dataset structure:", type(train_dataset))
            if hasattr(train_dataset, 'element_spec'):
                st.write("Debug: Dataset element spec:", train_dataset.element_spec)

            for x, y in train_dataset.take(1):
                st.write("Debug: Sample batch shapes - X:", x.shape, "y:", y.shape)
            # Train model
            callbacks = self.create_callbacks(self.model.name)
            history = self.model.train(
                train_dataset,
                val_dataset,
                epochs=epochs,
                callbacks=callbacks
            )

            # Evaluate
            test_loss = self.model.evaluate(test_dataset)
            test_mae = test_loss[1] if isinstance(test_loss, list) else test_loss

            # Plot history
            self.plot_training_history(history)

            return test_mae

        except Exception as e:
            st.error(f"Error in model training: {str(e)}")
            return None

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