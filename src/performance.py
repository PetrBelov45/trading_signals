"""
Performance Tracking Module - Track and audit true accuracy over time
Supports SQLite and JSONL storage
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
from pathlib import Path

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Track signal performance and calculate true accuracy over time
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        jsonl_path: Optional[str] = None
    ):
        """
        Initialize performance tracker

        Args:
            db_path: Path to SQLite database (optional)
            jsonl_path: Path to JSONL file (optional)
        """
        self.db_path = db_path
        self.jsonl_path = jsonl_path

        if db_path:
            self._init_database()

    def _init_database(self):
        """Initialize SQLite database"""
        try:
            import sqlite3

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    action TEXT NOT NULL,
                    p_up REAL,
                    confidence REAL,
                    expected_return REAL,
                    uncertainty REAL,
                    news_score REAL,
                    quality_score REAL,
                    current_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create outcomes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id INTEGER NOT NULL,
                    ticker TEXT NOT NULL,
                    horizon_days INTEGER NOT NULL,
                    actual_return REAL,
                    target_reached BOOLEAN,
                    stop_reached BOOLEAN,
                    correct_prediction BOOLEAN,
                    evaluated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (signal_id) REFERENCES signals(id)
                )
            ''')

            # Create indices
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ticker ON signals(ticker)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON signals(date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_outcome_ticker ON outcomes(ticker)')

            conn.commit()
            conn.close()

            logger.info(f"Initialized performance database: {self.db_path}")

        except Exception as e:
            logger.error(f"Database initialization error: {e}")

    def log_signal(
        self,
        ticker: str,
        signal: Dict[str, Any],
        quality_score: float,
        news_score: Optional[float] = None
    ):
        """
        Log a signal for future evaluation

        Args:
            ticker: Stock ticker
            signal: Signal dictionary
            quality_score: Quality score (0-1)
            news_score: News sentiment score (optional)
        """
        record = {
            'ticker': ticker,
            'date': datetime.now().isoformat(),
            'action': signal['action_raw'],
            'p_up': signal['p_up'],
            'confidence': signal['confidence'],
            'expected_return': signal['expected_return_pct'],
            'uncertainty': signal['uncertainty_pct'],
            'news_score': news_score,
            'quality_score': quality_score,
            'current_price': signal['current_price'],
            'stop_loss': signal['risk_plan'].get('stop_loss'),
            'take_profit': signal['risk_plan'].get('take_profit')
        }

        # Save to database
        if self.db_path:
            self._save_to_db(record)

        # Save to JSONL
        if self.jsonl_path:
            self._save_to_jsonl(record)

    def _save_to_db(self, record: Dict[str, Any]):
        """Save signal to SQLite database"""
        try:
            import sqlite3

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO signals (
                    ticker, date, action, p_up, confidence, expected_return,
                    uncertainty, news_score, quality_score, current_price,
                    stop_loss, take_profit
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record['ticker'],
                record['date'],
                record['action'],
                record['p_up'],
                record['confidence'],
                record['expected_return'],
                record['uncertainty'],
                record['news_score'],
                record['quality_score'],
                record['current_price'],
                record['stop_loss'],
                record['take_profit']
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Database save error: {e}")

    def _save_to_jsonl(self, record: Dict[str, Any]):
        """Append signal to JSONL file"""
        try:
            Path(self.jsonl_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.jsonl_path, 'a') as f:
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            logger.error(f"JSONL save error: {e}")

    def evaluate_signals(
        self,
        ticker: str,
        current_price: float,
        lookback_days: int = 30
    ):
        """
        Evaluate past signals and update outcomes

        Args:
            ticker: Stock ticker
            current_price: Current price
            lookback_days: Number of days to look back
        """
        if not self.db_path:
            return

        try:
            import sqlite3

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Find signals that need evaluation
            cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()

            cursor.execute('''
                SELECT s.id, s.ticker, s.date, s.action, s.current_price,
                       s.expected_return, s.stop_loss, s.take_profit
                FROM signals s
                LEFT JOIN outcomes o ON s.id = o.signal_id AND o.horizon_days = ?
                WHERE s.ticker = ? AND s.date >= ? AND o.id IS NULL
            ''', (lookback_days, ticker, cutoff_date))

            signals = cursor.fetchall()

            for signal in signals:
                signal_id, ticker, date, action, entry_price, exp_return, stop, target = signal

                # Calculate actual return
                actual_return = ((current_price - entry_price) / entry_price) * 100

                # Check if targets reached (simplified)
                target_reached = False
                stop_reached = False

                if action == "BUY":
                    if target and current_price >= target:
                        target_reached = True
                    if stop and current_price <= stop:
                        stop_reached = True
                elif action == "SELL":
                    if target and current_price <= target:
                        target_reached = True
                    if stop and current_price >= stop:
                        stop_reached = True

                # Correct prediction?
                correct = False
                if action == "BUY" and actual_return > 0:
                    correct = True
                elif action == "SELL" and actual_return < 0:
                    correct = True

                # Save outcome
                cursor.execute('''
                    INSERT INTO outcomes (
                        signal_id, ticker, horizon_days, actual_return,
                        target_reached, stop_reached, correct_prediction
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal_id,
                    ticker,
                    lookback_days,
                    actual_return,
                    target_reached,
                    stop_reached,
                    correct
                ))

            conn.commit()
            conn.close()

            logger.info(f"Evaluated {len(signals)} signals for {ticker}")

        except Exception as e:
            logger.error(f"Signal evaluation error: {e}")

    def get_true_accuracy(
        self,
        ticker: str,
        lookback_days: int = 180
    ) -> Dict[str, Any]:
        """
        Calculate true accuracy metrics for ticker

        Args:
            ticker: Stock ticker
            lookback_days: Number of days to look back

        Returns:
            Dictionary with accuracy metrics
        """
        if not self.db_path:
            return {'true_accuracy': 0.5, 'trades_count': 0}

        try:
            import sqlite3

            conn = sqlite3.connect(self.db_path)

            # Get outcomes
            query = '''
                SELECT o.correct_prediction, o.actual_return, o.target_reached,
                       o.stop_reached, s.action
                FROM outcomes o
                JOIN signals s ON o.signal_id = s.id
                WHERE o.ticker = ?
                  AND o.evaluated_at >= datetime('now', '-{} days')
            '''.format(lookback_days)

            df = pd.read_sql_query(query, conn, params=(ticker,))
            conn.close()

            if df.empty or len(df) < 5:
                return {'true_accuracy': 0.5, 'trades_count': len(df)}

            # Calculate metrics
            hit_rate = df['correct_prediction'].mean()

            returns = df['actual_return'].values
            avg_win = returns[returns > 0].mean() if np.any(returns > 0) else 0
            avg_loss = returns[returns < 0].mean() if np.any(returns < 0) else 0

            # Sharpe-like metric
            if returns.std() > 0:
                sharpe_norm = np.clip((returns.mean() / returns.std() + 1) / 3, 0, 1)
            else:
                sharpe_norm = 0.5

            # Max drawdown approximation
            cumulative = np.cumprod(1 + returns / 100)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_dd = abs(np.min(drawdown)) * 100

            max_dd_norm = 1 - np.clip(max_dd / 20, 0, 1)

            # Edge
            edge = avg_win + avg_loss  # avg_loss is negative
            edge_norm = np.clip((edge + 2) / 4, 0, 1)

            # True accuracy composite
            true_accuracy = (
                0.4 * hit_rate +
                0.2 * sharpe_norm +
                0.2 * max_dd_norm +
                0.2 * edge_norm
            )

            return {
                'true_accuracy': float(true_accuracy),
                'trades_count': len(df),
                'lookback_days': lookback_days,
                'hit_rate': float(hit_rate),
                'avg_win_pct': float(avg_win),
                'avg_loss_pct': float(avg_loss),
                'max_drawdown_pct': float(max_dd)
            }

        except Exception as e:
            logger.error(f"True accuracy calculation error: {e}")
            return {'true_accuracy': 0.5, 'trades_count': 0}

    def get_confusion_matrix(
        self,
        ticker: str,
        horizons: List[int] = [30, 90, 180]
    ) -> Dict[int, Dict[str, int]]:
        """
        Get confusion matrices for different horizons

        Args:
            ticker: Stock ticker
            horizons: List of horizons to analyze (days)

        Returns:
            Dictionary mapping horizon to confusion matrix
        """
        if not self.db_path:
            return {}

        try:
            import sqlite3

            conn = sqlite3.connect(self.db_path)
            results = {}

            for horizon in horizons:
                query = '''
                    SELECT s.action, o.correct_prediction
                    FROM outcomes o
                    JOIN signals s ON o.signal_id = s.id
                    WHERE o.ticker = ? AND o.horizon_days = ?
                '''

                df = pd.read_sql_query(query, conn, params=(ticker, horizon))

                if not df.empty:
                    tp = len(df[(df['action'] == 'BUY') & (df['correct_prediction'] == 1)])
                    fp = len(df[(df['action'] == 'BUY') & (df['correct_prediction'] == 0)])
                    tn = len(df[(df['action'] == 'SELL') & (df['correct_prediction'] == 1)])
                    fn = len(df[(df['action'] == 'SELL') & (df['correct_prediction'] == 0)])

                    results[horizon] = {
                        'true_positives': tp,
                        'false_positives': fp,
                        'true_negatives': tn,
                        'false_negatives': fn
                    }

            conn.close()
            return results

        except Exception as e:
            logger.error(f"Confusion matrix calculation error: {e}")
            return {}
