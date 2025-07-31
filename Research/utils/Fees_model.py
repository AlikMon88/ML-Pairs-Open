''' Fees Modeller for Spot/Margin and Perp. Futures '''

import pandas as pd
import numpy as np

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
    def get_funding_rate_history(self, client, symbols, start_str, end_str):
        
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
        start_ms = int(start_str.timestamp() * 1000)
        end_ms = int(end_str.timestamp() * 1000)

        for symbol in symbols:
            symbol = "".join(symbol.split('/'))
            print(f"  - Fetching for {symbol}")
            # The API returns a max of 1000 records per call, so we loop if needed
            # For simplicity, this example assumes one call is enough. A robust version
            # would loop until all data in the date range is retrieved.
            data = client.futures_funding_rate(
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
        
        # --- Aggregation to Daily Frequency ---
        df['date'] = df['fundingTime'].dt.date
        daily_funding_sum = df.groupby(['symbol', 'date'])['fundingRate'].sum().reset_index()
        
        # Convert date column back to datetime objects for proper indexing
        daily_funding_sum['date'] = pd.to_datetime(daily_funding_sum['date'])
        
        # 1. Create the full daily date range for the backtest
        full_date_range = pd.date_range(start=start_str, end=end_str, freq='D')
        
        # 2. Get the unique list of symbols we have data for
        symbols = df['symbol'].unique()
        
        # 3. Create the full cartesian product of symbols and dates
        multi_index = pd.MultiIndex.from_product(
            [symbols, full_date_range], 
            names=['symbol', 'date']
        )
        
        # 4. Create the final DataFrame with this complete index
        final_df = pd.DataFrame(index=multi_index)

        # 5. Set the index of our calculated data to match for merging
        daily_funding_sum.set_index(['symbol', 'date'], inplace=True)
        
        # 6. Join our calculated sums onto the complete index
        final_df = final_df.join(daily_funding_sum)
        
        # 7. Fill missing values with 0. Days with no funding events have zero cost.
        final_df['fundingRate'].fillna(0, inplace=True)
        
        # The result is a long-format, multi-indexed DataFrame ready for use.
        print("Funding data successfully aggregated to daily and multi-indexed.")
        return final_df
        
    
if __name__ == '__main__':
    print('Trading-Fees-Estimator (Binance-EX/HyperLiquid) ...')