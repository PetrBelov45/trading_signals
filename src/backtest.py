"""
Walk-Forward Backtesting Module
Purged time-series splits to avoid look-ahead bias
"""
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .models import MLTradingModel, prepare_features_labels


logger = logging.getLogger(__name__)


class WalkForwardBacktest:
    """
    Walk-forward backtesting with purged splits
    Train on historical window, test on next period, roll forward
    """

    def __init__(
        self,
        n_splits: int = 5,
        lookback_days: int = 500,
        horizon: int = 3,
        transaction_cost: float = 0.001,  # 10 bps
        slippage: float = 0.0008  # 8 bps
    ):
        """
        Initialize walk-forward backtester

        Args:
            n_splits: Number of walk-forward splits
            lookback_days: Days of history to train on
            horizon: Prediction horizon (days)
            transaction_cost: Transaction cost (as decimal, e.g., 0.001 = 10 bps)
            slippage: Slippage (as decimal)
        """
        self.n_splits = n_splits
        self.lookback_days = lookback_days
        self.horizon = horizon
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.total_costs = transaction_cost + slippage

    def create_splits(
        self,
        df: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create walk-forward splits

        Args:
            df: Full dataset with features and labels

        Returns:
            List of (train_df, test_df) tuples
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")

        total_days = (df.index[-1] - df.index[0]).days
        test_days = (total_days - self.lookback_days) // self.n_splits

        if test_days < self.horizon:
            logger.warning(f"Test period ({test_days} days) shorter than horizon ({self.horizon})")
            test_days = self.horizon

        splits = []

        for i in range(self.n_splits):
            # Define test period
            test_start_idx = self.lookback_days + (i * test_days)
            test_end_idx = test_start_idx + test_days

            # Ensure indices are within bounds
            if test_end_idx > len(df):
                test_end_idx = len(df)

            # Train on all data before test period
            train_df = df.iloc[:test_start_idx]
            test_df = df.iloc[test_start_idx:test_end_idx]

            if len(train_df) < 100 or len(test_df) < self.horizon:
                logger.warning(f"Split {i+1}: insufficient data (train={len(train_df)}, test={len(test_df)})")
                continue

            splits.append((train_df, test_df))
            logger.info(
                f"Split {i+1}: Train [{train_df.index[0].date()} to {train_df.index[-1].date()}] "
                f"Test [{test_df.index[0].date()} to {test_df.index[-1].date()}]"
            )

        return splits

    def run_backtest(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        model_type: str = 'random_forest',
        calibrate: bool = False
    ) -> Dict[str, Any]:
        """
        Run walk-forward backtest

        Args:
            df: Full dataset with features and labels
            feature_cols: List of feature columns
            model_type: Model type to use
            calibrate: Whether to calibrate probabilities

        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting walk-forward backtest: {self.n_splits} splits")

        splits = self.create_splits(df)

        if not splits:
            logger.error("No valid splits created")
            return {}

        all_predictions = []
        all_actuals = []
        all_probas = []
        split_metrics = []

        for i, (train_df, test_df) in enumerate(splits):
            logger.info(f"Running split {i+1}/{len(splits)}")

            # Prepare data
            X_train, y_train = prepare_features_labels(train_df, feature_cols)
            X_test, y_test = prepare_features_labels(test_df, feature_cols)

            # Train model
            model = MLTradingModel(
                model_type=model_type,
                task='classification',
                calibrate=calibrate,
                random_state=42 + i
            )

            metrics = model.fit(X_train, y_train, X_test, y_test)

            # Predict on test set
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            all_predictions.extend(y_pred)
            all_actuals.extend(y_test.values)
            all_probas.extend(y_proba[:, 1])  # Probability of positive class

            split_metrics.append(metrics.get('val', {}))

        # Calculate aggregate metrics
        results = self._calculate_metrics(
            np.array(all_actuals),
            np.array(all_predictions),
            np.array(all_probas),
            df.loc[df.index.isin([df.index[j] for j, _ in enumerate(all_predictions)])]
        )

        results['split_metrics'] = split_metrics
        results['n_splits'] = len(splits)

        return results

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate comprehensive backtest metrics"""

        # Classification metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Trading metrics
        # Simulate trades based on predictions
        returns = self._simulate_trades(y_pred, df)

        if len(returns) > 0:
            total_return = np.sum(returns)
            win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0
            avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0
            avg_loss = np.mean(returns[returns < 0]) if np.any(returns < 0) else 0

            # Risk-adjusted metrics
            sharpe = self._calculate_sharpe(returns)
            sortino = self._calculate_sortino(returns)
            max_dd = self._calculate_max_drawdown(returns)

            # CAGR
            days = (df.index[-1] - df.index[0]).days
            years = days / 365.25
            cagr = ((1 + total_return) ** (1 / years) - 1) * 100 if years > 0 else 0
        else:
            total_return = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            sharpe = 0
            sortino = 0
            max_dd = 0
            cagr = 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'hitrate': win_rate,
            'total_return_pct': total_return * 100,
            'cagr_pct': cagr,
            'avg_win_pct': avg_win * 100,
            'avg_loss_pct': avg_loss * 100,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown_pct': max_dd * 100,
            'n_trades': len(returns),
            'turnover': len(returns) / len(y_pred) if len(y_pred) > 0 else 0
        }

    def _simulate_trades(self, predictions: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        """Simulate trades based on predictions"""
        returns = []

        for i, pred in enumerate(predictions):
            if i >= len(df):
                break

            if pred == 1:  # BUY signal
                # Get actual return over horizon
                if i + self.horizon < len(df):
                    actual_return = (
                        df['close'].iloc[i + self.horizon] / df['close'].iloc[i] - 1
                    )
                    # Subtract costs
                    net_return = actual_return - self.total_costs
                    returns.append(net_return)

        return np.array(returns)

    def _calculate_sharpe(self, returns: np.ndarray, risk_free: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0:
            return 0

        mean_return = np.mean(returns) * 252  # Annualized
        std_return = np.std(returns) * np.sqrt(252)  # Annualized

        if std_return == 0:
            return 0

        return (mean_return - risk_free) / std_return

    def _calculate_sortino(self, returns: np.ndarray, risk_free: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0

        mean_return = np.mean(returns) * 252
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            return 0

        downside_std = np.std(downside_returns) * np.sqrt(252)

        if downside_std == 0:
            return 0

        return (mean_return - risk_free) / downside_std

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0

        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        return np.min(drawdown)


def calculate_true_accuracy(
    backtest_results: Dict[str, Any],
    weights: Dict[str, float] = None
) -> float:
    """
    Calculate composite true accuracy score from backtest results

    Args:
        backtest_results: Results from walk-forward backtest
        weights: Custom weights for each component

    Returns:
        True accuracy score (0-1)
    """
    if weights is None:
        weights = {
            'hitrate': 0.4,
            'sharpe': 0.2,
            'max_dd': 0.2,
            'edge': 0.2
        }

    # Normalize components to 0-1
    hitrate_norm = backtest_results.get('hitrate', 0.5)

    sharpe = backtest_results.get('sharpe', 0)
    sharpe_norm = np.clip((sharpe + 1) / 3, 0, 1)  # Normalize Sharpe (-1 to 2) to (0-1)

    max_dd = backtest_results.get('max_drawdown_pct', 0)
    max_dd_norm = 1 - np.clip(abs(max_dd) / 20, 0, 1)  # Lower DD is better

    # Edge (avg win - avg loss, normalized)
    avg_win = backtest_results.get('avg_win_pct', 0)
    avg_loss = backtest_results.get('avg_loss_pct', 0)
    edge = avg_win + avg_loss  # avg_loss is negative
    edge_norm = np.clip((edge + 2) / 4, 0, 1)  # Normalize to 0-1

    true_accuracy = (
        weights['hitrate'] * hitrate_norm +
        weights['sharpe'] * sharpe_norm +
        weights['max_dd'] * max_dd_norm +
        weights['edge'] * edge_norm
    )

    return np.clip(true_accuracy, 0, 1)
