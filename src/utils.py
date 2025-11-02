"""
Utility functions for ML Trading Signal Generator
"""
import os
import json
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timedelta
import logging

import numpy as np
import pandas as pd


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass


def cache_key(*args) -> str:
    """Generate cache key from arguments"""
    key_str = "_".join(str(arg) for arg in args)
    return hashlib.md5(key_str.encode()).hexdigest()


class DiskCache:
    """Simple disk-based cache with TTL"""

    def __init__(self, cache_dir: str = ".cache", ttl: int = 900):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = ttl

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached dataframe if exists and not expired"""
        cache_file = self.cache_dir / f"{key}.parquet"
        if not cache_file.exists():
            return None

        # Check TTL
        age = time.time() - cache_file.stat().st_mtime
        if age > self.ttl:
            cache_file.unlink()
            return None

        try:
            return pd.read_parquet(cache_file)
        except Exception:
            return None

    def set(self, key: str, df: pd.DataFrame):
        """Cache dataframe to disk"""
        cache_file = self.cache_dir / f"{key}.parquet"
        try:
            df.to_parquet(cache_file)
        except Exception:
            pass


def align_to_trading_calendar(df: pd.DataFrame, timezone: str = "UTC") -> pd.DataFrame:
    """Ensure dataframe is aligned to trading days only"""
    if df.empty:
        return df

    # Remove duplicates and sort
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()

    return df


def clean_numeric_data(df: pd.DataFrame, fill_method: str = 'ffill') -> pd.DataFrame:
    """Clean numeric data - handle NaN, inf values"""
    df = df.copy()

    # Replace inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Forward fill then backward fill
    if fill_method == 'ffill':
        df = df.fillna(method='ffill').fillna(method='bfill')
    elif fill_method == 'drop':
        df = df.dropna()

    return df


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV dataframe to standard column names"""
    # Map various column name formats to standard names
    column_map = {
        'Date': 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Adj Close': 'close',  # Prefer adjusted close
        'Volume': 'volume',
        'TRADEDATE': 'date',
        'OPEN': 'open',
        'HIGH': 'high',
        'LOW': 'low',
        'CLOSE': 'close',
        'VOLUME': 'volume',
    }

    df = df.copy()
    df.columns = [column_map.get(col, col.lower()) for col in df.columns]

    # Ensure we have required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Set date as index if not already
    if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    return df[required]


def save_json(data: Dict[str, Any], filepath: str, pretty: bool = True):
    """Save data to JSON file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        if pretty:
            json.dump(data, f, indent=2, default=str)
        else:
            json.dump(data, f, default=str)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def append_jsonl(data: Dict[str, Any], filepath: str):
    """Append data to JSONL file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'a') as f:
        f.write(json.dumps(data, default=str) + '\n')


def load_watchlist(filepath: str) -> list:
    """Load ticker watchlist from file"""
    with open(filepath, 'r') as f:
        tickers = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return tickers


def calculate_position_size(confidence: float, min_pct: float = 5.0, max_pct: float = 15.0) -> float:
    """Calculate position size based on confidence"""
    # Scale linearly from min to max based on confidence
    return min_pct + (max_pct - min_pct) * confidence


def format_currency(value: float) -> str:
    """Format value as currency"""
    return f"${value:,.2f}"


def format_percent(value: float, decimals: int = 2) -> str:
    """Format value as percentage"""
    return f"{value:.{decimals}f}%"


def get_bar_chart(value: float, max_value: float = 1.0, width: int = 10) -> str:
    """Generate simple ASCII bar chart"""
    filled = int((value / max_value) * width)
    return "█" * filled + "░" * (width - filled)


def days_ago(days: int) -> datetime:
    """Get datetime N days ago"""
    return datetime.now() - timedelta(days=days)


def retry_with_backoff(func, max_retries: int = 4, base_delay: float = 2.0):
    """Retry function with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
    return None
