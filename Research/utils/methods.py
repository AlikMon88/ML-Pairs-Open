import pandas
from statsmodels.tsa.stattools import coint
import numpy as np

def SelectCointegratedPairs(symbols, cluster_labels, history, hurst_threshold = 0.5):
        
        pairs = {}
        clusters = {}
        
        # Group symbols by clusters
        for symbol, label in zip(symbols, cluster_labels):
            clusters.setdefault(label, []).append(symbol)
        
        print(' ---- Clusters ---- ')
        print()
        print(clusters)
        print()
        print(' ------------- ')

        # For each cluster, test for cointegration
        for cluster_id, cluster_symbols in clusters.items():
            if len(cluster_symbols) < 2:
                continue
            
            print(f"Testing cointegration in cluster {cluster_id} with {len(cluster_symbols)} symbols.")
            
            for i in range(len(cluster_symbols)):
                for j in range(i+1, len(cluster_symbols)):
                    
                    sym_i = cluster_symbols[i]
                    sym_j = cluster_symbols[j]

                    print(sym_i, sym_j)

                    try:
                        prices_i = history[sym_i]["TRDPRC_1"]
                        prices_j = history[sym_j]["TRDPRC_1"]
                        
                        if prices_i.isnull().any() or prices_j.isnull().any():
                            print(f"Missing data for pair {sym_i}-{sym_j}. Skipping.")
                            continue
                        
                        coint_t, pvalue, _ = coint(prices_i, prices_j)
                        print('p-Value: ', pvalue)

                        if pvalue < 0.05:
                            
                            # Calculate hedge ratio using rolling regression
                            hedge_ratio = CalculateHedgeRatio(prices_j, prices_i)
                            
                            # Calculate spread
                            spread = prices_i - hedge_ratio * prices_j
                            spread_mean, spread_std = CalculateRollingSpreadStats(spread)
                            
                            # Calculate Hurst Exponent
                            hurst = CalculateHurstExponent(spread)
                            if hurst >= hurst_threshold:
                                print(f"Pair {sym_i}-{sym_j} rejected due to Hurst exponent {hurst:.2f}")
                                continue  # Not mean-reverting
                            
                            pair_key = f"{sym_i}_{sym_j}"
                            pairs[pair_key] = {
                                'symbol_y': sym_i,
                                'symbol_x': sym_j,
                                'hedge_ratio': hedge_ratio,
                                'spread_mean': spread_mean,
                                'spread_std': spread_std,
                                'hurst': hurst,
                                'spread_series': spread  # Store spread series
                            }
                            print(f"Cointegrated pair found: {pair_key} with p-value {pvalue} and Hurst {hurst:.2f}")
                    except Exception as e:
                        print(f"Error testing pair {sym_i}-{sym_j}: {e}")
                        continue
        return pairs
    
def CalculateHedgeRatio(x, y):
    # Rolling window regression to calculate dynamic hedge ratio
    window = 60  # 60 days rolling window
    if len(x) < window:
        window = len(x)
    try:
        hedge_ratio = np.polyfit(x[-window:], y[-window:], 1)[0]
        return hedge_ratio
    except:
        return 1.0  # Default hedge ratio if regression fails

def CalculateRollingSpreadStats(spread):
    # Calculate rolling mean and std using EMA
    lambda_val = 0.94
    spread_series = spread.values
    mu = spread_series[0]
    sigma = spread_series.std()
    for s in spread_series:
        mu = lambda_val * mu + (1 - lambda_val) * s
        sigma = lambda_val * sigma + (1 - lambda_val) * (s - mu)**2
    sigma = np.sqrt(sigma)
    return mu, sigma

def CalculateHurstExponent(time_series):
    # Simple Hurst exponent calculation
    N = len(time_series)
    if N < 20:
        return 0.5  # Default value
    lag_max = min(20, N//2)
    tau = []
    for lag in range(2, lag_max):
        pp = np.polyfit(range(len(time_series)), time_series, 1)
        trend = np.polyval(pp, range(len(time_series)))
        detrended = time_series - trend
        lag_detrended = detrended[lag:]
        r = np.max(lag_detrended) - np.min(lag_detrended)
        s = np.std(detrended)
        if s != 0:
            tau.append(r/s)
    if len(tau) == 0:
        return 0.5
    hurst = np.polyfit(np.log(range(2, lag_max)), np.log(tau), 1)[0]
    return hurst

def FilterHighCorrelationPairs(pairs):
    # Calculate correlation between spreads and filter out highly correlated pairs
    pair_keys = list(pairs.keys())
    
    # Create a DataFrame with spread time series
    spread_dict = {pair: pairs[pair]['spread_series'] for pair in pair_keys}
    
    spreads_df = pd.DataFrame(spread_dict)
    if spreads_df.empty:
        return pairs  # No pairs to filter
    
    # Compute the correlation matrix
    corr_matrix = spreads_df.corr()
    
    # Identify pairs to remove
    pairs_to_remove = set()
    for i in range(len(pair_keys)):
        for j in range(i + 1, len(pair_keys)):
            pair_i = pair_keys[i]
            pair_j = pair_keys[j]
            if pair_i in pairs_to_remove or pair_j in pairs_to_remove:
                continue  # Already marked for removal
            correlation = corr_matrix.loc[pair_i, pair_j]
            if correlation > correlation_threshold:
                # Exclude the pair with higher spread_std
                if pairs[pair_i]['spread_std'] > pairs[pair_j]['spread_std']:
                    Debug(f"Excluding pair {pair_j} due to high correlation ({correlation:.2f}) with {pair_i}")
                    pairs_to_remove.add(pair_j)
                else:
                    Debug(f"Excluding pair {pair_i} due to high correlation ({correlation:.2f}) with {pair_j}")
                    pairs_to_remove.add(pair_i)
    
    # Remove the identified pairs
    for pair in pairs_to_remove:
        del pairs[pair]
    
    return pairs

def CalculateMarketVolatility():
    # Calculate rolling volatility of SPY using EMA
    history = History(spy, market_volatility_window, Resolution.Daily)
    if history.empty:
        return market_volatility
    returns = history['close'].pct_change().dropna()
    if len(returns) == 0:
        return market_volatility
    ema_variance = returns.ewm(span=market_volatility_window).var().iloc[-1]
    ema_std = np.sqrt(ema_variance)
    return ema_std

def ManagePositions():
    # Calculate market volatility scaling factor
    if market_volatility > 0:
        scaling_factor = 1.0 / market_volatility
    else:
        scaling_factor = 1.0  # Default scaling
    
    # Monitor open positions and generate signals
    for pair_key, position in open_positions.items():
        symbol_y = position['symbol_y']
        symbol_x = position['symbol_x']
        hedge_ratio = position['hedge_ratio']
        spread_mean = position['spread_mean']
        spread_std = position['spread_std']
        
        # Check if data is available
        if not (Securities.ContainsKey(symbol_y) and Securities.ContainsKey(symbol_x)):
            continue
        price_y = Securities[symbol_y].Price
        price_x = Securities[symbol_x].Price
        
        if price_y == 0 or price_x == 0:
            continue
        
        # Calculate current spread and z-score
        spread = price_y - hedge_ratio * price_x
        z_score = (spread - spread_mean) / spread_std
        
        # Adjust z-score for transaction costs
        adjusted_z = abs(z_score) - transaction_cost * 100  # Normalize transaction cost
        
        # Entry signal
        if not position['is_open']:
            if adjusted_z > entry_threshold:
                # Position sizing
                allocation = scaling_factor * (1.0 / spread_std) / len(pairs)
                allocation = min(allocation, 0.05)  # Max 5% per position
                
                # Go long spread: short x, long y
                if z_score > entry_threshold:
                    SetHoldings(symbol_y, allocation)
                    SetHoldings(symbol_x, -allocation * hedge_ratio)
                    Debug(f"Entering long spread for pair {pair_key} at z-score {z_score:.2f}")
                # Go short spread: long x, short y
                elif z_score < -entry_threshold:
                    SetHoldings(symbol_y, -allocation)
                    SetHoldings(symbol_x, allocation * hedge_ratio)
                    Debug(f"Entering short spread for pair {pair_key} at z-score {z_score:.2f}")
                position['is_open'] = True
                position['entry_z'] = z_score
        else:
            # Exit conditions
            exit_condition = abs(z_score) < exit_threshold
            stop_loss_condition = abs(z_score) > stop_loss
            take_profit_condition = (take_profit > 0) and (abs(z_score) > take_profit)
            
            if exit_condition or stop_loss_condition or take_profit_condition:
                Liquidate(symbol_y)
                Liquidate(symbol_x)
                Debug(f"Exiting position for pair {pair_key} at z-score {z_score:.2f}")
                position['is_open'] = False

def LiquidatePositionsNotInPairs(self):
    # Liquidate positions that are not in the current pairs
    current_symbols = set()
    for pair_info in pairs.values():
        current_symbols.add(pair_info['symbol_y'])
        current_symbols.add(pair_info['symbol_x'])
    
    for kvp in Portfolio:
        symbol = kvp.Key
        if symbol not in current_symbols and Portfolio[symbol].Invested:
            Liquidate(symbol)
            Debug(f"Liquidating position for symbol {symbol.Value} not in current pairs.")

def GetFundamentalData(self, symbol):
    # Retrieve fundamental data (P/E and P/B ratios) for the given symbol
    if symbol in Securities:
        security = Securities[symbol]
        if security.HasData and security.Fundamentals is not None:
            pe_ratio = security.Fundamentals.ValuationRatios.PERatio
            pb_ratio = security.Fundamentals.ValuationRatios.PBRatio
            if pe_ratio is not None and pb_ratio is not None:
                return {'PE': pe_ratio, 'PB': pb_ratio}
    return None

def GetSectorCode(self, symbol):
    # Get sector code for the symbol
    if symbol in Securities:
        security = Securities[symbol]
        if security.HasData and security.Fundamentals is not None:
            sector_code = security.Fundamentals.AssetClassification.MorningstarSectorCode
            return sector_code
    return 0


if __name__ == '__main__':
    print('Running __main.py__')