import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def preprocess_data(file_path, sequence_length=60):
    """
    Preprocesses the data for the LSTM model.

    Args:
        file_path (str): The path to the CSV file with historical data.
        sequence_length (int): The number of time steps to look back.

    Returns:
        tuple: A tuple containing scaled data, scaler object, training sequences, and targets.
    """
    print("Preprocessing data...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")

    # Load data
    data = pd.read_csv(file_path)
    
    # Ensure 'Close' column contains only numeric values
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data = data.dropna(subset=['Close'])
    close_prices = data['Close'].values.reshape(-1, 1)

    # Scale the data to be between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Create sequences
    X = []
    y = []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    
    # Reshape data for LSTM model (samples, timesteps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    print("Data preprocessing complete.")
    return scaled_data, scaler, X, y

if __name__ == '__main__':
    TICKER_SYMBOL = 'DOGE-USD'
    FILE_PATH = f'../data/{TICKER_SYMBOL}_historical_data.csv'
    
    # Preprocess the data
    scaled_data, scaler, X_train, y_train = preprocess_data(FILE_PATH)
    
    print(f"Shape of training data (X): {X_train.shape}")
    print(f"Shape of target data (y): {y_train.shape}")
    print("Data preprocessing complete.")