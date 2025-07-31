''' Fees Modeller for Spot/Margin and Perp. Futures '''

import pandas as pd
import numpy as np
from binance.client import Client

class FeesModel():
    def __init__(self):
        pass
    
    def spot_margin_binance(self):
        pass
    
    def perp_futures_binance(self):
        pass

    def perp_futures_hyperliquid(self):
        pass
        
        
    ### Get Funding-Fees
    def get_funding_rate_history(client, symbols, start_str, end_str):
        
        """
        Fetches historical funding rate data for multiple symbols and formats it
        into a single wide-format DataFrame.

        Args:
            client: An initialized Binance client.
            symbols: A list of asset tickers, e.g., ['BTCUSDT', 'ETHUSDT'].
            start_str: The start date in string format, e.g., "2021-01-01".
            end_str: The end date in string format, e.g., "2023-12-31".

        Returns:
            pd.DataFrame: A DataFrame with timestamps as the index and a column for
                        each symbol's funding rate.
        """
        
        print("Fetching historical funding rates...")
        all_funding_data = []
        
        # Convert string dates to milliseconds for the API
        start_ms = start_str
        end_ms = end_str

        for symbol in symbols:
            print(f"  - Fetching for {symbol}")
            # The API returns a max of 1000 records per call, so we loop if needed
            # For simplicity, this example assumes one call is enough. A robust version
            # would loop until all data in the date range is retrieved.
            data = client.funding_rate_history(
                symbol=symbol, 
                startTime=start_ms, 
                endTime=end_ms, 
                limit=1000
            )
            all_funding_data.extend(data)

        if not all_funding_data:
            print("No funding data found for the given symbols/period.")
            return pd.DataFrame()

        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(all_funding_data)
        
        # --- Data Cleaning and Formatting ---
        # Convert types
        df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
        df['fundingRate'] = pd.to_numeric(df['fundingRate'])

        # Keep only the columns we need
        df = df[['symbol', 'fundingTime', 'fundingRate']]
        
        # Pivot the DataFrame to get the desired format
        # Index = timestamp, Columns = symbols
        pivoted_df = df.pivot(index='fundingTime', columns='symbol', values='fundingRate')
        
        # Reindex to match your hourly market data, and forward-fill the values
        # The funding rate is constant between funding events.
        full_date_range = pd.date_range(start=start_str, end=end_str, freq='H') # Assuming hourly data
        pivoted_df = pivoted_df.reindex(full_date_range).ffill()
        
        # Fill any remaining NaNs at the beginning with 0
        pivoted_df.fillna(0, inplace=True)

        print("Funding data successfully loaded and formatted.")
        return pivoted_df
    
if __name__ == '__main__':
    print('Trading-Fees-Estimator (Binance-EX/HyperLiquid) ...')