import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

def predict_future(ticker, days_to_predict=30, sequence_length=60):
    """
    Loads a trained model and predicts future prices.

    Args:
        ticker (str): The ticker symbol for which to predict.
        days_to_predict (int): The number of future days to predict.
        sequence_length (int): The sequence length used during training.
    """
    # Construct paths relative to the script's location
    script_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(script_dir, '..', 'models', f'{ticker}_predictor_model.h5')
    data_path = os.path.join(script_dir, '..', 'data', f'{ticker}_historical_data.csv')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
        
    model = load_model(model_path)
    data = pd.read_csv(data_path)
    
    # Prepare the scaler
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Get the last 'sequence_length' days from the historical data
    last_sequence = scaled_data[-sequence_length:]
    current_batch = last_sequence.reshape(1, sequence_length, 1)
    
    future_predictions = []
    
    print(f"Predicting the next {days_to_predict} days...")
    for _ in range(days_to_predict):
        # Predict the next price
        next_prediction = model.predict(current_batch)[0]
        future_predictions.append(next_prediction)
        
        # Update the batch to include the new prediction and remove the oldest value
        current_batch = np.append(current_batch[:, 1:, :], [[next_prediction]], axis=1)

    # Inverse transform the predictions to get actual price values
    predicted_prices = scaler.inverse_transform(future_predictions)

    print("Future Predicted Prices:")
    print(predicted_prices)

    # --- Visualization ---
    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'], data['Close'], color='blue', label='Historical Prices')
    
    # Create future dates for plotting
    last_date = pd.to_datetime(data['Date'].iloc[-1])
    future_dates = pd.to_datetime([last_date + pd.DateOffset(days=x) for x in range(1, days_to_predict + 1)])
    
    plt.plot(future_dates, predicted_prices, color='red', linestyle='--', label='Predicted Prices')
    plt.title(f'{ticker} Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    TICKER_SYMBOL = 'BTC-USD'
    predict_future(TICKER_SYMBOL, days_to_predict=30)
    print("Future price prediction complete.")