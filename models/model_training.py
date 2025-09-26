import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from data_preprocessing import preprocess_data

def build_model(input_shape):
    """
    Builds the LSTM model architecture.

    Args:
        input_shape (tuple): The shape of the input data.

    Returns:
        keras.Model: The compiled LSTM model.
    """
    model = Sequential()
    
    # First LSTM layer with Dropout
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    # Second LSTM layer with Dropout
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Dense layer
    model.add(Dense(units=25))
    
    # Output layer
    model.add(Dense(units=1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("Model built successfully.")
    model.summary()
    return model

def train_model(model, X_train, y_train, ticker, epochs=25, batch_size=32):
    """
    Trains the LSTM model.

    Args:
        model (keras.Model): The model to train.
        X_train (np.array): The training feature data.
        y_train (np.array): The training target data.
        ticker (str): The ticker symbol for naming the saved model.
        epochs (int): The number of training epochs.
        batch_size (int): The size of batches for training.
    """
    print("Starting model training...")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    # Create models directory if it doesn't exist
    if not os.path.exists('../models'):
        os.makedirs('../models')
        
    # Save the model
    model_path = f'../models/{ticker}_predictor_model.h5'
    model.save(model_path)
    print(f"Model trained and saved to {model_path}")
    return history

if __name__ == '__main__':
    # --- Configuration ---
    TICKER_SYMBOL = 'BTC-USD'
    FILE_PATH = f'../data/{TICKER_SYMBOL}_historical_data.csv'
    
    # 1. Preprocess the data
    scaled_data, scaler, X_train, y_train = preprocess_data(FILE_PATH)
    
    # 2. Build the model
    # The input shape is (sequence_length, num_features) which is (60, 1)
    model = build_model(input_shape=(X_train.shape[1], 1))
    
    # 3. Train the model
    train_model(model, X_train, y_train, TICKER_SYMBOL)