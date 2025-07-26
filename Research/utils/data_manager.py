# src/data_manager.py
import pandas as pd
from binance.client import Client
import os
from config import BINANCE_API_KEY, BINANCE_API_SECRET, DATA_DIR

class DataManager:
    def __init__(self):
        self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

    def download_data(self, symbol, interval, start_str, end_str):
        """Downloads historical kline data and saves it as a parquet file."""
        filepath = os.path.join(DATA_DIR, f"{symbol}-{interval}.parquet")
        print(f"Downloading {symbol} data...")
        klines = self.client.get_historical_klines(symbol, interval, start_str, end_str)
        
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_asset_volume', 'number_of_trades', 
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        
        # Convert columns to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.to_parquet(filepath)
        print(f"Data for {symbol} saved to {filepath}")

    def load_data(self, symbol, interval):
        """Loads data from a parquet file."""
        filepath = os.path.join(DATA_DIR, f"{symbol}-{interval}.parquet")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}. Please download first.")
        return pd.read_parquet(filepath)