# import refinitiv.data as rd
import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
from IPython.display import display 

''' Interactive Plots '''
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def _manual_universe_creation():
    tickers = ["AAPL.OQ", "MSFT.OQ", "GOOGL.OQ", "AMZN.OQ", "META.OQ", "TSLA.OQ", 
               "NVDA.OQ", "JNJ", "AMD.OQ", "INTC.OQ", "QCOM.OQ", "ADBE.OQ", "NFLX.OQ", 
               "PYPL.OQ", "SBUX.OQ", "INTU.OQ", "TEAM.OQ", "BIDU.OQ", "EXPE.OQ"]
    return tickers

### Sector-wise? Return ordering based on internal ref listing 
def _call_stocks(limit = 30, is_nasdaq = False):
    sp500_constituents = rd.get_data(
        universe=["0#.SPX"],  # Chain RIC for S&P 500 constituents
        fields=["TR.IndexConstituentRIC", "TR.IndexConstituentName"],
        parameters={"SDate": "0"}  # "0" for latest constituents
    )
    df_cons = pd.DataFrame(sp500_constituents)[:limit]
    stocks = np.array(df_cons['Instrument']).tolist()

    if is_nasdaq:
        stocks = [_s for _s in stocks if _s.split('.')[-1] == 'OQ']

    print(' ---> Number of Sampled Stocks: ', len(stocks))

    return stocks

if __name__ == '__main__':
    rd.open_session()
    print('Running __retrieve.py__ now ...')
    df_stocks = _call_stocks()
    print(df_stocks.head())
    rd.close_session()
    
    
''' 4 yr default (365 trading days for crypto (24/7)) | NEED: add crypto fundamentals'''
''' load_universe=True | We pick 50 most liquid tokens for physical universe-selection (from Binance Exchange)'''
def get_ccxt_crypto_data(timeframe='1d', limit=1460, is_plot=False, load_universe=False):
    
    # 1. Fetch Data using CCXT
    exchange = ccxt.binance()  # Using Binance as the source
    if not exchange.has['fetchOHLCV']:
        print(f"The selected exchange ({exchange.id}) does not support fetching OHLCV data.")
        return
    
    # Load market data
    markets = exchange.load_markets()

    # Filter symbols with USDT pairs
    usdt_markets = [symbol for symbol in markets if symbol.endswith('/USDT')][:100]
    print('USDT-markets-retrieved: ', len(usdt_markets), usdt_markets[:2])

    # Fetch tickers (includes volume info)
    tickers = exchange.fetch_tickers(usdt_markets)

    # Sort by quote volume in descending order
    sorted_markets = sorted(
        tickers.items(),
        key=lambda x: x[1]['quoteVolume'] if x[1]['quoteVolume'] is not None else 0,
        reverse=True
    )

    # Top 50 most liquid tokens
    symbols = [ticker for ticker, _ in sorted_markets[:50]]
    print(symbols)
    
    if not load_universe:
        symbols = [symbols[0]]
        
    
    candle_dict =  {}
    for symbol in symbols:
        
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not ohlcv:
            print(f"Could not fetch data for {symbol}. The symbol may be invalid for {exchange.id}.")
            return

        candle_dict[symbol] = ohlcv

    df_dict = {}    
    for symbol, ohlcvs in candle_dict.items():
        df = pd.DataFrame(ohlcvs,  columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        df.drop(columns='timestamp', inplace=True)
        df_dict[symbol] = df 
    
    df_universe = pd.concat(df_dict, axis=0)
    display(df_universe.head(5))
    
    # df_universe.columns.names = ['symbol', 'ohlcv'] ## level 0 and level 1 col names for multi-index dataframes
    # df_universe.index.name = 'datetime'
        
    if is_plot:

        # 3. Create Interactive Plot
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{symbol} Candlestick', 'Volume'),
            row_width=[0.2, 0.7]
        )

        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="OHLC"
        ), row=1, col=1)

        # Volume chart
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color='orange'
        ), row=2, col=1)

        fig.update_layout(
            title_text=f"{symbol} Price Data",
            xaxis_rangeslider_visible=False
        )
        print(f"Displaying chart for {symbol}...")
        fig.show()

    return df_universe, symbols

''' Use 'Plotly' to create interactive plots | 1008 trading days / 255 per year '''
def get_yfinance_equities_data(ticker='AAPL', period="4y", is_plot=False):
    
    # 1. Fetch Data using yfinance
    stock = yf.Ticker(ticker)
    hist_df = stock.history(period=period)
    info = stock.info

    if hist_df.empty:
        print(f"Error: Could not fetch historical data for {ticker}. It may be delisted or an invalid ticker.")
        return

    # 2. Prepare Fundamental Data for Display
    market_cap = info.get('marketCap', 'N/A')
    pe_ratio = info.get('trailingPE', 'N/A')
    sector = info.get('sector', 'N/A')
    summary = info.get('longBusinessSummary', 'No summary available.')
    
    fund_df = pd.DataFrame({'mkcp': [market_cap], 'pe_ratio': [pe_ratio], 'sector': [sector], 'summary': [summary]})

    if is_plot:

        # 3. Create Interactive Plot
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{ticker} Candlestick', 'Volume'),
            row_width=[0.2, 0.7]
        )

        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=hist_df.index,
            open=hist_df['Open'],
            high=hist_df['High'],
            low=hist_df['Low'],
            close=hist_df['Close'],
            name="OHLC"
        ), row=1, col=1)

        # Volume chart
        fig.add_trace(go.Bar(
            x=hist_df.index,
            y=hist_df['Volume'],
            name='Volume',
            marker_color='royalblue'
        ), row=2, col=1)


        fig.update_layout(
            title_text=f"{ticker} Price Data",
            xaxis_rangeslider_visible=False
        )        

        print(f"Displaying chart for {ticker}...")
        fig.show()
        
    
    return hist_df, fund_df

