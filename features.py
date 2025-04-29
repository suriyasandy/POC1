import pandas as pd
import numpy as np

def validate_annualization(df: pd.DataFrame, vol_col: str = 'Volatility') -> pd.DataFrame:
    """
    Ensure Volatility is annualized; if not, scale by sqrt(252).
    Business note: annualized volatility allows consistent thresholding
    across different time windows.
    """
    df = df.copy()
    if df[vol_col].max() < 0.5:
        df[vol_col] *= np.sqrt(252)
    return df

def compute_log_returns(df: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
    """
    Compute daily log returns to capture relative price moves.
    Business note: log returns are symmetric and scale-free,
    unlike simple differences.
    """
    df = df.copy()
    df['LogReturn'] = np.log(df[price_col] / df[price_col].shift(1))
    return df.dropna()

def add_technical_indicators(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Add:
      - RSI: momentum indicator
      - MACD: trend/momentum crossover
      - Bollinger Bands: dynamic price bands
    Business note: these help explain why volatility spikes,
    giving traders context beyond raw σ.
    """
    df = df.copy()

    # RSI
    delta    = df['LogReturn'].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs       = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD & Signal Line
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    m = df['Close'].rolling(window).mean()
    s = df['Close'].rolling(window).std()
    df['BB_Upper'] = m + 2 * s
    df['BB_Lower'] = m - 2 * s

    return df.dropna()

def create_rolling_and_lag_features(
    df: pd.DataFrame,
    vol_col: str = 'Volatility',
    return_col: str = 'LogReturn',
    windows: list[int] = [7, 14, 30, 60]
) -> pd.DataFrame:
    """
    Build rolling means & std-devs, lags, and seasonality.
    Business note: captures evolving volatility regimes,
    autocorrelation, and calendar effects.
    """
    df = df.copy()
    for w in windows:
        df[f'Vol_MA_{w}']  = df[vol_col].rolling(w).mean()
        df[f'Vol_STD_{w}'] = df[vol_col].rolling(w).std()
        df[f'Ret_MA_{w}']  = df[return_col].rolling(w).mean()
        df[f'Ret_STD_{w}'] = df[return_col].rolling(w).std()

    df['Vol_Lag1'] = df[vol_col].shift(1)
    df['Ret_Lag1'] = df[return_col].shift(1)
    df['Ret_Lag7'] = df[return_col].shift(7)

    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month']     = df['Date'].dt.month

    return df.dropna()

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline: validate → log returns → technicals → rolling/lag.
    """
    df = df.copy()
    df = validate_annualization(df)
    df = compute_log_returns(df)
    df = add_technical_indicators(df)
    df = create_rolling_and_lag_features(df)
    return df.reset_index(drop=True)
