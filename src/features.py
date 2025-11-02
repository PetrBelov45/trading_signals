"""
Feature Engineering Module - NO LEAKAGE GUARANTEES
All features at time t use only data <= t
"""
import logging
from typing import Optional

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering with strict no-leakage guarantees.
    All features at time t are computed using only data up to and including t.
    """

    def __init__(self, add_calendar_features: bool = False):
        self.add_calendar_features = add_calendar_features
        self.feature_names = []

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all features from OHLCV data

        Args:
            df: DataFrame with OHLCV data (index must be datetime)

        Returns:
            DataFrame with engineered features (NaN rows dropped)
        """
        df = df.copy()

        logger.info(f"Engineering features for {len(df)} bars")

        # Price-based features
        df = self._add_returns(df)
        df = self._add_momentum(df)

        # Trend indicators
        df = self._add_moving_averages(df)
        df = self._add_trend_slope(df)

        # Momentum oscillators
        df = self._add_rsi(df)
        df = self._add_macd(df)
        df = self._add_stochastic(df)

        # Volatility indicators
        df = self._add_atr(df)
        df = self._add_realized_volatility(df)

        # Bollinger Bands
        df = self._add_bollinger_bands(df)

        # Volume indicators
        df = self._add_volume_features(df)

        # Calendar features (optional)
        if self.add_calendar_features:
            df = self._add_calendar_features(df)

        # Clean up
        df = self._clean_features(df)

        # Track feature names (exclude OHLCV)
        self.feature_names = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]

        logger.info(f"Generated {len(self.feature_names)} features, {len(df)} bars after cleanup")

        return df

    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return features (1, 3, 5, 10, 20 day returns)"""
        for period in [1, 3, 5, 10, 20]:
            df[f'return_{period}d'] = df['close'].pct_change(period)

        return df

    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features (z-scores of returns)"""
        for period in [5, 10, 20]:
            returns = df['close'].pct_change(period)
            mean = returns.rolling(window=20, min_periods=10).mean()
            std = returns.rolling(window=20, min_periods=10).std()
            df[f'return_{period}d_zscore'] = (returns - mean) / (std + 1e-8)

        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving average features"""
        # Simple moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(window=period, min_periods=period).mean()
            df[f'price_to_sma_{period}'] = df['close'] / (df[f'sma_{period}'] + 1e-8)

        # Exponential moving averages
        for period in [5, 10, 20]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False, min_periods=period).mean()
            df[f'price_to_ema_{period}'] = df['close'] / (df[f'ema_{period}'] + 1e-8)

        return df

    def _add_trend_slope(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend slope features"""
        for period in [20, 50]:
            # Linear regression slope over window
            def compute_slope(series):
                if len(series) < period:
                    return np.nan
                x = np.arange(len(series))
                y = series.values
                slope = np.polyfit(x, y, 1)[0]
                return slope / (y[-1] + 1e-8)  # Normalize by last price

            df[f'slope_{period}'] = df['close'].rolling(window=period, min_periods=period).apply(compute_slope, raw=False)

        return df

    def _add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()

        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))

        # Normalized RSI
        df['rsi_norm'] = (df['rsi'] - 50) / 50

        return df

    def _add_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Add MACD indicators"""
        ema_fast = df['close'].ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False, min_periods=slow).mean()

        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False, min_periods=signal).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Normalize by price
        df['macd_norm'] = df['macd'] / (df['close'] + 1e-8)
        df['macd_hist_norm'] = df['macd_hist'] / (df['close'] + 1e-8)

        return df

    def _add_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Add Stochastic Oscillator"""
        low_min = df['low'].rolling(window=k_period, min_periods=k_period).min()
        high_max = df['high'].rolling(window=k_period, min_periods=k_period).max()

        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-8)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period, min_periods=d_period).mean()

        # Normalized (0-1 scale)
        df['stoch_k_norm'] = df['stoch_k'] / 100
        df['stoch_d_norm'] = df['stoch_d'] / 100

        return df

    def _add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=period, min_periods=period).mean()

        # ATR as percentage of close
        df['atr_pct'] = 100 * df['atr'] / (df['close'] + 1e-8)

        return df

    def _add_realized_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realized volatility features"""
        returns = df['close'].pct_change()

        for period in [10, 20]:
            # Annualized volatility
            df[f'realized_vol_{period}'] = returns.rolling(window=period, min_periods=period).std() * np.sqrt(252)

        return df

    def _add_bollinger_bands(self, df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """Add Bollinger Bands"""
        sma = df['close'].rolling(window=period, min_periods=period).mean()
        std = df['close'].rolling(window=period, min_periods=period).std()

        df['bb_upper'] = sma + (std * num_std)
        df['bb_lower'] = sma - (std * num_std)
        df['bb_middle'] = sma

        # Bollinger Band width (normalized)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-8)

        # Position within bands
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        # Volume moving average
        df['volume_sma_20'] = df['volume'].rolling(window=20, min_periods=20).mean()

        # Volume ratio
        df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-8)

        # Volume momentum
        df['volume_momentum_5'] = df['volume'].pct_change(5)

        # Money flow (price * volume)
        df['money_flow'] = df['close'] * df['volume']
        df['money_flow_ratio'] = df['money_flow'] / df['money_flow'].rolling(window=20, min_periods=20).mean()

        return df

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar features (weekday, month)"""
        if isinstance(df.index, pd.DatetimeIndex):
            df['weekday'] = df.index.dayofweek
            df['month'] = df.index.month

            # One-hot encode weekday
            for i in range(5):  # Monday=0 to Friday=4
                df[f'weekday_{i}'] = (df['weekday'] == i).astype(int)

        return df

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean features - replace inf with NaN and drop NaN rows"""
        # Replace inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Drop rows with NaN (usually warm-up period)
        rows_before = len(df)
        df = df.dropna()
        rows_after = len(df)

        if rows_before > rows_after:
            logger.info(f"Dropped {rows_before - rows_after} warm-up rows with NaN values")

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """Get list of feature columns (excluding OHLCV)"""
        exclude = ['open', 'high', 'low', 'close', 'volume']
        return [col for col in df.columns if col not in exclude]


def create_labels(
    df: pd.DataFrame,
    horizon: int = 3,
    threshold: float = 0.2,
    label_type: str = 'binary'
) -> pd.DataFrame:
    """
    Create labels for ML training - with proper shift to avoid leakage

    Args:
        df: DataFrame with features
        horizon: Number of days ahead to predict
        threshold: Threshold for classification (in %)
        label_type: 'binary', 'ternary', or 'regression'

    Returns:
        DataFrame with 'label' column added
    """
    df = df.copy()

    # Calculate future return (shift by -horizon to look ahead)
    future_return = df['close'].pct_change(horizon).shift(-horizon) * 100  # in %

    if label_type == 'binary':
        # 1 if return > threshold, else 0
        df['label'] = (future_return > threshold).astype(int)

    elif label_type == 'ternary':
        # BUY (2), NEUTRAL (1), SELL (0)
        df['label'] = 1  # Default neutral
        df.loc[future_return > threshold, 'label'] = 2  # BUY
        df.loc[future_return < -threshold, 'label'] = 0  # SELL

    elif label_type == 'regression':
        # Direct future return
        df['label'] = future_return

    else:
        raise ValueError(f"Unknown label_type: {label_type}")

    # Drop rows where label is NaN (last 'horizon' rows)
    df = df.dropna(subset=['label'])

    logger.info(f"Created {label_type} labels with horizon={horizon}, threshold={threshold}%")

    return df
