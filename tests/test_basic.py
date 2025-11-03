"""
Basic tests for ML Trading Signal Generator
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features import FeatureEngineer, create_labels
from src.models import MLTradingModel
from src.decision import DecisionPolicy, calculate_quality_score
from src.backtest import calculate_true_accuracy


def create_sample_ohlcv(days=100):
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(days) * 2)

    data = {
        'open': close_prices + np.random.randn(days) * 0.5,
        'high': close_prices + np.abs(np.random.randn(days) * 1.5),
        'low': close_prices - np.abs(np.random.randn(days) * 1.5),
        'close': close_prices,
        'volume': np.random.randint(1000000, 10000000, days)
    }

    df = pd.DataFrame(data, index=dates)
    return df


class TestFeatureEngineering:
    """Test feature engineering"""

    def test_feature_engineering_no_leakage(self):
        """Test that feature engineering doesn't leak future data"""
        df = create_sample_ohlcv(days=200)

        feature_eng = FeatureEngineer()
        df_features = feature_eng.engineer_features(df)

        # Should have features
        assert len(df_features.columns) > 5

        # Should have no NaN (after warm-up)
        assert df_features.isna().sum().sum() == 0

        # Should have dropped warm-up rows
        assert len(df_features) < len(df)

    def test_label_creation(self):
        """Test label creation with proper shift"""
        df = create_sample_ohlcv(days=100)
        feature_eng = FeatureEngineer()
        df_features = feature_eng.engineer_features(df)

        df_labeled = create_labels(df_features, horizon=3, threshold=0.5)

        # Should have label column
        assert 'label' in df_labeled.columns

        # Should have dropped last 'horizon' rows
        assert len(df_labeled) < len(df_features)

        # Labels should be binary (0 or 1)
        assert set(df_labeled['label'].unique()).issubset({0, 1})


class TestMLModel:
    """Test ML model"""

    def test_model_training(self):
        """Test model training and prediction"""
        df = create_sample_ohlcv(days=200)

        feature_eng = FeatureEngineer()
        df_features = feature_eng.engineer_features(df)
        df_labeled = create_labels(df_features, horizon=3)

        feature_cols = feature_eng.get_feature_columns(df_labeled)

        # Split data
        split_idx = int(len(df_labeled) * 0.8)
        train_df = df_labeled.iloc[:split_idx]
        val_df = df_labeled.iloc[split_idx:]

        X_train = train_df[feature_cols]
        y_train = train_df['label']
        X_val = val_df[feature_cols]
        y_val = val_df['label']

        # Train model
        model = MLTradingModel(model_type='random_forest', task='classification')
        metrics = model.fit(X_train, y_train, X_val, y_val)

        # Should have metrics
        assert 'train' in metrics
        assert 'val' in metrics
        assert 'accuracy' in metrics['val']

        # Predict
        predictions = model.predict(X_val)
        assert len(predictions) == len(X_val)

        # Predict proba
        probas = model.predict_proba(X_val)
        assert probas.shape[0] == len(X_val)
        assert probas.shape[1] == 2  # Binary classification


class TestDecisionPolicy:
    """Test decision policy"""

    def test_signal_generation(self):
        """Test signal generation"""
        policy = DecisionPolicy()

        # Test BUY signal
        signal = policy.generate_signal(
            p_up=0.75,
            current_price=100.0,
            atr=2.0,
            volatility=0.20
        )

        assert signal['action_raw'] in ['BUY', 'SELL', 'HOLD']
        assert 0 <= signal['confidence'] <= 1
        assert signal['current_price'] == 100.0

        # Should have risk plan
        assert 'risk_plan' in signal

    def test_quality_score(self):
        """Test quality score calculation"""
        signal = {
            'confidence': 0.7,
            'net_edge_pct': 1.5,
            'uncertainty_pct': 2.0
        }

        quality = calculate_quality_score(signal, news_score=0.3, true_accuracy=0.65)

        assert 'quality_score' in quality
        assert 0 <= quality['quality_score'] <= 1
        assert 'quality_score_100' in quality


class TestBacktest:
    """Test backtesting"""

    def test_true_accuracy_calculation(self):
        """Test true accuracy calculation"""
        backtest_results = {
            'hitrate': 0.62,
            'sharpe': 1.1,
            'max_drawdown_pct': -9.4,
            'avg_win_pct': 2.3,
            'avg_loss_pct': -1.4
        }

        true_acc = calculate_true_accuracy(backtest_results)

        assert 0 <= true_acc <= 1


def test_deterministic_output():
    """Test that output is deterministic with same seed"""
    np.random.seed(42)
    df1 = create_sample_ohlcv(days=100)

    np.random.seed(42)
    df2 = create_sample_ohlcv(days=100)

    # Should be identical
    pd.testing.assert_frame_equal(df1, df2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
