import pandas as pd
import yfinance as yf
import pandas_ta as ta
import numpy as np
from fredapi import Fred

FRED_API_KEY = '88f89b6134c3f03d05307558eeeafcf7'


def dl_data(name: str, start_date: str) -> pd.DataFrame:
    df = yf.download(name, start=start_date)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
    df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']

    df['real_logs'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Close'].pct_change().rolling(21).std()
    df['Z_Score_10'] = (df['Close'] - df['Close'].rolling(10).mean()) / df['Close'].rolling(10).std()

    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['SMA_200'] = ta.sma(df['Close'], length=200)
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_10'] = ta.sma(df['Close'], length=10)

    df['Dist_SMA_50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
    df['Dist_SMA_200'] = (df['Close'] - df['SMA_200']) / df['SMA_200']
    df['Dist_SMA_10'] = (df['Close'] - df['SMA_10']) / df['SMA_10']

    df['Log_Volume'] = np.log(df['Volume'] + 1)
    df['Log_Vol_Change'] = df['Log_Volume'].diff().shift(1)
    df['Trend_Signal'] = (df['Close'] > df['SMA_10']).astype(int)

    df['Ret_L1'] = df['Close'].pct_change().shift(1)
    df['Ret_L2'] = df['Close'].pct_change().shift(2)

    vix = yf.download("^VIX", start=start_date)['Close']
    df['VIX'] = vix.shift(1)
    df['VIX_Change'] = df['VIX'].diff()

    try:
        fred = Fred(api_key=FRED_API_KEY)
        yield_10y = fred.get_series('DGS10', observation_start=start_date)
        yield_2y = fred.get_series('DGS2', observation_start=start_date)
        fed_rate = fred.get_series('FEDFUNDS', observation_start=start_date)
        unemployment = fred.get_series('UNRATE', observation_start=start_date)
        inflation_idx = fred.get_series('CPIAUCSL', observation_start=start_date)
        stress_fed = fred.get_series('STLFSI4', observation_start=start_date)

        macro_df = pd.DataFrame({
            'Fed_Rate': fed_rate,
            'Unemployment': unemployment,
            'Inflation_Idx': inflation_idx,
            'stress_fed': stress_fed,
            'yield_10y': yield_10y,
            'yield_2y': yield_2y
        })
        macro_df.index = pd.to_datetime(macro_df.index)

        df = df.merge(macro_df, left_index=True, right_index=True, how='left')

        df[['Fed_Rate', 'Unemployment', 'Inflation_Idx', 'stress_fed', 'yield_10y', 'yield_2y']] = df[
            ['Fed_Rate', 'Unemployment', 'Inflation_Idx', 'stress_fed', 'yield_10y', 'yield_2y']].ffill().shift(30)

        df['Inflation_var'] = df['Inflation_Idx'].pct_change(periods=252)
        df['Yield_Spread'] = df['yield_10y'] - df['yield_2y']
        df['Yield_Spread_Change'] = df['Yield_Spread'].diff()

    except Exception as e:
        print(f"Error: {e}")

    df['target_1'] = (df['Close'].shift(-1) > df["Close"]).astype(int)
    df['target_2'] = np.log(df["Close"].shift(-1) / df["Close"])
    df['5j'] = df['real_logs'].rolling(window=5).sum().shift(-5)
    df['target_3'] = (df['5j'] > 0).astype(int)

    df.dropna(inplace=True)
    return df


def preprocess_data(data: pd.DataFrame, features: list, target_num: int):
    cols_to_fix = ['RSI', 'MFI', 'ADX', 'Volatility', 'Z_Score_10', 'Dist_SMA_10', 'Dist_SMA_50', 'Dist_SMA_200']
    data[cols_to_fix] = data[cols_to_fix].shift(1)
    data.dropna(inplace=True)

    X = data[features]
    market_logs = data[["real_logs"]]

    if target_num == 1:
        y = data[["target_1"]]
    elif target_num == 2:
        y = data[["target_2"]]
    elif target_num == 3:
        y = data[["target_3"]]
    else:
        raise ValueError("Invalid target_num")

    split_val = int(len(data) * 0.65)
    split_test = int(len(data) * 0.85)

    X_train = X.iloc[:split_val]
    X_val = X.iloc[split_val:split_test]
    X_test = X.iloc[split_test:]

    y_train = y.iloc[:split_val]
    y_val = y.iloc[split_val:split_test]
    y_test = y.iloc[split_test:]

    market_logs_test = market_logs.iloc[split_test:]
    market_logs_val = market_logs.iloc[split_val:split_test]

    return X_train, X_val, X_test, y_train, y_val, y_test, market_logs_test, market_logs_val