# Time Series Forecasting Application

A comprehensive Streamlit application for time series forecasting using various models including traditional statistical methods, machine learning, and deep learning approaches.

## Features

- Multiple model support:
  - Traditional Models (ARIMA, SARIMA)
  - Machine Learning Models (Random Forest, XGBoost)
  - Deep Learning Models (RNN, LSTM, Stacked LSTM+RNN)
- Interactive data visualization
- Model comparison and evaluation
- Prediction intervals
- Cross-validation support
- Feature engineering
- Model diagnostics

## Installation

1. Clone the repository:
```bash
cd Streamlit_ML_team_app
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run app/main.py
```

The application will be available at `http://localhost:8501`.

### Data Input

The application supports multiple data input methods:
- Upload CSV files
- Load from GitHub repository
- Use example datasets

### Model Selection

1. Choose model categories:
   - Traditional Models
   - Machine Learning Models
   - Deep Learning Models

2. Configure model parameters:
   - Sequence length for deep learning models
   - Number of estimators for tree-based models
   - ARIMA/SARIMA orders
   - Learning rates and other hyperparameters

### Training and Evaluation

The application provides:
- Cross-validation results
- Model performance metrics
- Visualization of predictions
- Model diagnostics
- Prediction intervals

## Project Structure

```
project/
├── app/
│   ├── main.py               # Main Streamlit application
│   ├── config.py            # Configuration and constants
│   ├── data/                # Data handling
│   ├── models/              # Model implementations
│   └── utils/               # Utility functions
├── data/                    # Sample datasets
├── docs/                    # Documentation
├── tests/                   # Unit tests
├── requirements.txt
└── README.md
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
The project follows PEP 8 style guide. Format code using:
```bash
black .
isort .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Inspired by various time series forecasting techniques and best practices

## Authors

- 

## Contact

- Email:
