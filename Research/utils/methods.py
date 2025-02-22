import pandas
from statsmodels.tsa.stattools import coint
import numpy as np
import pandas as pd

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

                    try:
                        prices_i = history[sym_i]["TRDPRC_1"]
                        prices_j = history[sym_j]["TRDPRC_1"]
                        
                        if prices_i.isnull().any() or prices_j.isnull().any():
                            print(f"Missing data for pair {sym_i}-{sym_j}. Skipping.")
                            continue
                        
                        coint_t, pvalue, _ = coint(prices_i, prices_j)

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

if __name__ == '__main__':
    print('Running __main.py__')