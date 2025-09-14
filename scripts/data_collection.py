import yfinance as yf
import pandas as pd
import os

def fetch_data(ticker, start_date, end_date):
    """
    Fetches historical market data from Yahoo Finance.

    Args:
        ticker (str): The ticker symbol (e.g., 'BTC-USD' for Bitcoin, 'AAPL' for Apple).
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame containing the historical data.
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Create the data directory if it doesn't exist
    if not os.path.exists('../data'):
        os.makedirs('../data')
        
    # Save the data to a CSV file
    file_path = f'../data/{ticker}_historical_data.csv'
    data.to_csv(file_path)
    print(f"Data saved to {file_path}")
    
    return data

if __name__ == '__main__':
    # --- Configuration ---
    # For crypto, use format like 'BTC-USD', 'ETH-USD'
    # For stocks, use format like 'AAPL', 'GOOGL'
    TICKER_SYMBOL = 'BTC-USD' 
    START_DATE = '2020-01-01'
    END_DATE = '2025-09-14' # Today's date
    
    fetch_data(TICKER_SYMBOL, START_DATE, END_DATE)