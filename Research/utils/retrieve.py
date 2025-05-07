# import refinitiv.data as rd
import pandas as pd
import numpy as np

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