model_params = {}
rnn_params = {}
decision_tree_params = {}
stacked_params = {}

if advanced_settings:
    if model_choice == "RNN":
        st.subheader("Advanced RNN Settings")
        rnn_params.update({
            "num_layers" = st.sidebar.slider("Number of layers", 1, 10, 1),
            "num_units" = st.sidebar.slider("Number of units", 1, 128, 16),
            "activation": st.selectbox("Activation function", ["relu", "tanh", "sigmoid"]),
            "optimizer": st.selectbox("Optimizer", ["adam", "sgd", "rmsprop"]),
            "learning_rate": st.number_input("Learning rate (optional)", 0.0001, 1.0, step=0.0001, value=0.001, format="%.4f"),
            "look_back": st.number_input("Look-back window size", 1, 365, value=3),
            "epochs": st.number_input("Number of epochs", 1, 1000, value=200),
            "batch_size": st.number_input("Batch size", 1, 128, value=32),
            "recurrent_dropout": st.number_input("Recurrent dropout rate", 0.0, 0.5, step=0.1, value=0.1, format="%.1f"),
        })
    elif model_choice == "Decision Tree":
        st.subheader("Advanced Decision Tree Settings")
        xgboost_params.update({
            "lags": st.number_input("Number of lags", 1, 365, value=12),
            "max_depth": st.number_input("Max depth", 1, 20, value=3),
            "learning_rate": st.number_input("Learning rate", 0.0001, 1.0, step=0.0001, value=0.1, format="%.4f"),
            "n_estimators": st.number_input("Number of estimators", 1, 1000, value=100),
            "min_samples_split": st.number_input("Min samples split", 2, 10, value=2),
            })
    elif model_choice == "Stacked":
        st.subheader("Advanced Stacked Deep Learning Model Settings")
        stacked_params.update({
            "num_layers" = st.sidebar.slider("Number of layers", 1, 10, 1),
            "num_units" = st.sidebar.slider("Number of units", 1, 128, 16),
            "activation": st.selectbox("Activation function", ["relu", "tanh", "sigmoid"]),
            "optimizer": st.selectbox("Optimizer", ["adam", "rmsprop"]),
            "learning_rate": st.number_input("Learning rate (optional)", 0.0001, 1.0, step=0.0001, value=0.001, format="%.4f"),
            "look_back": st.number_input("Look-back window size", 1, 365, value=3),
            "epochs": st.number_input("Number of epochs", 1, 200, value=100),
            "batch_size": st.number_input("Batch size", 1, 128, value=32),
            "recurrent_dropout": st.number_input("Recurrent dropout rate", 0.0, 0.5, step=0.1, value=0.1, format="%.1f"),
            "lags": st.number_input("Number of lags", 1, 365, value=12),
            "max_depth": st.number_input("Max depth", 1, 20, value=3),
            "learning_rate": st.number_input("Learning rate", 0.0001, 1.0, step=0.0001, value=0.1, format="%.4f")
        })



        