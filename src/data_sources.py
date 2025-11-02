"""
Data ingestion module - fetches OHLCV data from various sources
Supports: Yahoo Finance, MOEX, Alpha Vantage, CSV
"""
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import time

import pandas as pd
import numpy as np
import requests

from .utils import (
    cache_key, DiskCache, normalize_ohlcv,
    align_to_trading_calendar, clean_numeric_data, days_ago
)


logger = logging.getLogger(__name__)


class DataSource:
    """Base class for data sources"""

    def __init__(self, cache_ttl: int = 900):
        self.cache = DiskCache(ttl=cache_ttl)
        self.cache_ttl = cache_ttl

    def fetch(self, ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for ticker"""
        raise NotImplementedError


class YahooFinanceSource(DataSource):
    """Yahoo Finance data source using yfinance"""

    def fetch(self, ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance"""
        cache_k = cache_key("yahoo", ticker, start_date.date(), end_date.date())
        cached = self.cache.get(cache_k)
        if cached is not None:
            logger.info(f"Cache hit for {ticker}")
            return cached

        try:
            import yfinance as yf
            logger.info(f"Fetching {ticker} from Yahoo Finance")

            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, auto_adjust=True)

            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return None

            # Normalize columns
            df.index.name = 'date'
            df = df.reset_index()
            df = normalize_ohlcv(df)

            self.cache.set(cache_k, df)
            return df

        except Exception as e:
            logger.error(f"Yahoo Finance error for {ticker}: {e}")
            return None


class AlphaVantageSource(DataSource):
    """Alpha Vantage data source (requires API key)"""

    def __init__(self, api_key: str, cache_ttl: int = 900):
        super().__init__(cache_ttl)
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    def fetch(self, ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch data from Alpha Vantage"""
        cache_k = cache_key("alphavantage", ticker, start_date.date(), end_date.date())
        cached = self.cache.get(cache_k)
        if cached is not None:
            logger.info(f"Cache hit for {ticker}")
            return cached

        try:
            logger.info(f"Fetching {ticker} from Alpha Vantage")

            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': ticker,
                'outputsize': 'full',
                'apikey': self.api_key
            }

            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if 'Time Series (Daily)' not in data:
                logger.warning(f"No data in Alpha Vantage response for {ticker}")
                return None

            # Parse data
            ts = data['Time Series (Daily)']
            records = []
            for date_str, values in ts.items():
                date = pd.to_datetime(date_str)
                if start_date <= date <= end_date:
                    records.append({
                        'date': date,
                        'open': float(values['1. open']),
                        'high': float(values['2. high']),
                        'low': float(values['3. low']),
                        'close': float(values['5. adjusted close']),
                        'volume': int(values['6. volume'])
                    })

            if not records:
                return None

            df = pd.DataFrame(records)
            df.set_index('date', inplace=True)
            df = df.sort_index()

            self.cache.set(cache_k, df)
            return df

        except Exception as e:
            logger.error(f"Alpha Vantage error for {ticker}: {e}")
            return None


class MOEXSource(DataSource):
    """MOEX ISS API data source for Russian stocks"""

    def __init__(self, cache_ttl: int = 900):
        super().__init__(cache_ttl)
        self.base_url = "https://iss.moex.com/iss"

    def fetch(self, ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch data from MOEX ISS API"""
        cache_k = cache_key("moex", ticker, start_date.date(), end_date.date())
        cached = self.cache.get(cache_k)
        if cached is not None:
            logger.info(f"Cache hit for {ticker}")
            return cached

        try:
            logger.info(f"Fetching {ticker} from MOEX")

            # MOEX ISS API endpoint for historical data
            url = f"{self.base_url}/history/engines/stock/markets/shares/securities/{ticker}.json"

            params = {
                'from': start_date.strftime('%Y-%m-%d'),
                'till': end_date.strftime('%Y-%m-%d'),
                'start': 0
            }

            all_records = []
            while True:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                if 'history' not in data or 'data' not in data['history']:
                    break

                columns = data['history']['columns']
                rows = data['history']['data']

                if not rows:
                    break

                # Find column indices
                col_map = {col: idx for idx, col in enumerate(columns)}
                required = ['TRADEDATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']

                if not all(col in col_map for col in required):
                    logger.error(f"Missing required columns for {ticker}")
                    break

                for row in rows:
                    if row[col_map['OPEN']] is not None:  # Skip rows with no price data
                        all_records.append({
                            'date': pd.to_datetime(row[col_map['TRADEDATE']]),
                            'open': float(row[col_map['OPEN']]),
                            'high': float(row[col_map['HIGH']]),
                            'low': float(row[col_map['LOW']]),
                            'close': float(row[col_map['CLOSE']]),
                            'volume': int(row[col_map['VOLUME']] or 0)
                        })

                # Check if more pages available
                if len(rows) < 100:  # MOEX returns max 100 rows per page
                    break

                params['start'] += 100
                time.sleep(0.2)  # Rate limiting

            if not all_records:
                logger.warning(f"No data returned for {ticker} from MOEX")
                return None

            df = pd.DataFrame(all_records)
            df.set_index('date', inplace=True)
            df = df.sort_index()

            # Convert to Moscow timezone if needed
            if df.index.tz is None:
                df.index = df.index.tz_localize('Europe/Moscow')

            self.cache.set(cache_k, df)
            return df

        except Exception as e:
            logger.error(f"MOEX error for {ticker}: {e}")
            return None


class CSVSource(DataSource):
    """CSV file data source (fallback)"""

    def __init__(self, csv_path: str, cache_ttl: int = 900):
        super().__init__(cache_ttl)
        self.csv_path = csv_path

    def fetch(self, ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Load data from CSV file"""
        try:
            logger.info(f"Loading {ticker} from CSV: {self.csv_path}")

            df = pd.read_csv(self.csv_path)
            df = normalize_ohlcv(df)

            # Filter date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]

            if df.empty:
                return None

            return df

        except Exception as e:
            logger.error(f"CSV error for {ticker}: {e}")
            return None


class DataFetcher:
    """Main data fetcher with fallback logic"""

    def __init__(
        self,
        universe: str = "global",
        alpha_vantage_key: Optional[str] = None,
        csv_path: Optional[str] = None,
        csv_ru_path: Optional[str] = None,
        cache_ttl: int = 900
    ):
        self.universe = universe
        self.cache_ttl = cache_ttl

        # Setup sources based on universe
        if universe == "russia":
            self.primary = MOEXSource(cache_ttl)
            self.fallback = CSVSource(csv_ru_path, cache_ttl) if csv_ru_path else None
        else:  # global or custom
            self.primary = YahooFinanceSource(cache_ttl)
            self.fallback = AlphaVantageSource(alpha_vantage_key, cache_ttl) if alpha_vantage_key else None
            self.csv_fallback = CSVSource(csv_path, cache_ttl) if csv_path else None

    def fetch_ticker(
        self,
        ticker: str,
        history_days: int = 750,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single ticker with fallback logic

        Returns:
            DataFrame with OHLCV data or None if all sources fail
        """
        if end_date is None:
            end_date = datetime.now()
        start_date = end_date - timedelta(days=history_days)

        # Try primary source
        df = self.primary.fetch(ticker, start_date, end_date)
        if df is not None and len(df) > 0:
            return self._validate_and_clean(df, ticker, history_days)

        logger.warning(f"Primary source failed for {ticker}, trying fallback")

        # Try fallback sources
        if self.fallback:
            df = self.fallback.fetch(ticker, start_date, end_date)
            if df is not None and len(df) > 0:
                return self._validate_and_clean(df, ticker, history_days)

        if hasattr(self, 'csv_fallback') and self.csv_fallback:
            df = self.csv_fallback.fetch(ticker, start_date, end_date)
            if df is not None and len(df) > 0:
                return self._validate_and_clean(df, ticker, history_days)

        logger.error(f"All sources failed for {ticker}")
        return None

    def _validate_and_clean(
        self,
        df: pd.DataFrame,
        ticker: str,
        min_days: int = 400
    ) -> Optional[pd.DataFrame]:
        """Validate and clean fetched data"""
        try:
            # Align to trading calendar
            df = align_to_trading_calendar(df)

            # Clean numeric data
            df = clean_numeric_data(df, fill_method='ffill')

            # Check minimum length
            if len(df) < min_days:
                logger.warning(
                    f"Insufficient data for {ticker}: {len(df)} days (min {min_days})"
                )
                return None

            # Basic validation
            if (df['close'] <= 0).any():
                logger.warning(f"Invalid prices found for {ticker}")
                df = df[df['close'] > 0]

            if (df['volume'] < 0).any():
                logger.warning(f"Invalid volume found for {ticker}")
                df = df[df['volume'] >= 0]

            return df

        except Exception as e:
            logger.error(f"Validation error for {ticker}: {e}")
            return None

    def fetch_multiple(
        self,
        tickers: List[str],
        history_days: int = 750,
        parallel: int = 1
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers

        Args:
            tickers: List of ticker symbols
            history_days: Number of days of history
            parallel: Number of parallel fetches (currently sequential)

        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}

        for ticker in tickers:
            logger.info(f"Fetching {ticker}...")
            df = self.fetch_ticker(ticker, history_days)

            if df is not None:
                results[ticker] = df
                logger.info(f"✓ {ticker}: {len(df)} bars")
            else:
                logger.warning(f"✗ {ticker}: failed to fetch")

        return results


# Default universes
DEFAULT_GLOBAL_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B",
    "V", "JPM", "WMT", "MA", "PG", "UNH", "HD", "DIS", "BAC", "XOM",
    "COST", "ABBV", "CRM", "NFLX", "AMD", "PFE", "KO", "ADBE", "CSCO",
    "TMO", "MRK", "ACN", "NKE", "PEP", "LLY", "AVGO", "TXN", "INTC"
]

DEFAULT_RUSSIA_UNIVERSE = [
    "SBER", "GAZP", "LKOH", "GMKN", "TATN", "ROSN", "NVTK", "PLZL",
    "TCSG", "ALRS", "MGNT", "POLY", "AFKS", "MOEX", "YNDX", "MTSS",
    "SNGS", "NLMK", "CHMF", "MAGN", "IRAO", "FEES", "AFLT", "VTBR"
]


def get_default_universe(universe_name: str) -> List[str]:
    """Get default ticker universe"""
    if universe_name == "global":
        return DEFAULT_GLOBAL_UNIVERSE
    elif universe_name == "russia":
        return DEFAULT_RUSSIA_UNIVERSE
    else:
        return []
