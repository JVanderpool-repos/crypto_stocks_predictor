# Crypto & Stocks Predictor

A machine learning project that uses LSTM neural networks to predict cryptocurrency and stock prices based on historical data from Yahoo Finance.

## 🚀 Features

- **Data Collection**: Automated fetching of historical price data using Yahoo Finance API
- **Data Preprocessing**: Advanced data cleaning, scaling, and sequence preparation for LSTM models
- **LSTM Model Training**: Deep learning model with dropout layers for price prediction
- **Future Price Prediction**: Generate predictions for specified number of days ahead
- **Visualization**: Interactive charts showing historical vs predicted prices
- **Multi-Asset Support**: Works with both cryptocurrencies (BTC-USD, ETH-USD) and stocks (AAPL, GOOGL, etc.)

## 📁 Project Structure

```
crypto predictor/
├── data/                    # Historical data storage (CSV files)
├── models/                  # Trained models and training scripts
│   └── model_training.py    # LSTM model architecture and training
├── notebooks/               # Jupyter notebooks for analysis
├── scripts/                 # Core functionality scripts
│   ├── data_collection.py   # Yahoo Finance data fetching
│   ├── data_preprocessing.py # Data cleaning and preparation
│   └── predict.py           # Price prediction and visualization
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🛠 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/JVanderpool-repos/crypto_stocks_predictor.git
   cd crypto_stocks_predictor
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   
   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1
   
   # Windows (Command Prompt)
   .venv\Scripts\activate.bat
   
   # Linux/macOS
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Usage

### 1. Data Collection

Collect historical data for any ticker symbol:

```bash
cd scripts
python data_collection.py
```

**Configuration in `data_collection.py`:**
```python
TICKER_SYMBOL = 'BTC-USD'    # For Bitcoin
# TICKER_SYMBOL = 'ETH-USD'  # For Ethereum  
# TICKER_SYMBOL = 'AAPL'     # For Apple stock
START_DATE = '2020-01-01'
END_DATE = '2025-09-26'      # Current date
```

### 2. Data Preprocessing

Prepare data for machine learning:

```bash
python data_preprocessing.py
```

This creates sequences of 60 time steps for LSTM training.

### 3. Model Training

Train the LSTM model:

```bash
cd ../models
python model_training.py
```

**Model Architecture:**
- 2 LSTM layers (50 units each) with dropout (0.2)
- Dense layer (25 units)
- Output layer (1 unit for price prediction)
- Adam optimizer with MSE loss

### 4. Make Predictions

Generate future price predictions:

```bash
cd ../scripts
python predict.py
```

This will:
- Load the trained model
- Predict prices for the next 30 days
- Display predictions and create visualization plots

## 🔧 Configuration

### Supported Assets

**Cryptocurrencies:**
- `BTC-USD` (Bitcoin)
- `ETH-USD` (Ethereum)
- `ADA-USD` (Cardano)
- `DOT-USD` (Polkadot)

**Stocks:**
- `AAPL` (Apple)
- `GOOGL` (Google)
- `MSFT` (Microsoft)
- `TSLA` (Tesla)

### Model Parameters

```python
# In model_training.py
SEQUENCE_LENGTH = 60    # Look back period (days)
EPOCHS = 25            # Training iterations
BATCH_SIZE = 32        # Training batch size

# In predict.py
DAYS_TO_PREDICT = 30   # Future prediction horizon
```

## 📈 Example Output

```
Fetching data for BTC-USD from 2020-01-01 to 2025-09-26...
Data saved to ../data/BTC-USD_historical_data.csv

Preprocessing data...
Data preprocessing complete.
Shape of training data (X): (1825, 60, 1)
Shape of target data (y): (1825,)

Model built successfully.
Training model...
Model trained and saved to ../models/BTC-USD_predictor_model.h5

Predicting the next 30 days...
Future Predicted Prices:
[[63547.23]
 [63891.45]
 [64203.12]
 ...]
```

## 🧠 Technical Details

### LSTM Architecture

The model uses Long Short-Term Memory (LSTM) networks specifically designed for time series prediction:

- **Sequence Length**: 60 days of historical data to predict the next day
- **Features**: Close price (normalized using MinMaxScaler)
- **Layers**: 
  - LSTM (50 units, return_sequences=True) + Dropout(0.2)
  - LSTM (50 units) + Dropout(0.2)  
  - Dense (25 units)
  - Dense (1 unit, output)

### Data Processing

1. **Normalization**: MinMaxScaler (0-1 range)
2. **Sequence Creation**: Sliding window of 60 time steps
3. **Train/Test Split**: Chronological split (80/20)

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.x
- pandas
- numpy  
- yfinance
- scikit-learn
- matplotlib

## ⚠️ Disclaimer

**This project is for educational and research purposes only. Cryptocurrency and stock predictions are highly speculative and should not be used as the sole basis for financial decisions. Always do your own research and consider consulting with financial advisors before making investment decisions.**

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Author**: JVanderpool-repos  
**Repository**: [crypto_stocks_predictor](https://github.com/JVanderpool-repos/crypto_stocks_predictor)

---

⭐ **Star this repository if you found it helpful!**