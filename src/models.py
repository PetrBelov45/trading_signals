"""
ML Models Module - Classification and Regression models
"""
import logging
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, mean_absolute_error, mean_squared_error
)


logger = logging.getLogger(__name__)


class MLTradingModel:
    """
    ML model for trading signal generation
    Supports classification (binary/multi-class) and regression
    """

    def __init__(
        self,
        model_type: str = 'random_forest',
        task: str = 'classification',
        calibrate: bool = False,
        tune: bool = False,
        random_state: int = 42
    ):
        """
        Initialize ML model

        Args:
            model_type: 'random_forest', 'gradient_boosting', or 'xgb'
            task: 'classification' or 'regression'
            calibrate: Whether to calibrate probabilities
            tune: Whether to perform hyperparameter tuning
            random_state: Random seed
        """
        self.model_type = model_type
        self.task = task
        self.calibrate = calibrate
        self.tune = tune
        self.random_state = random_state

        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.feature_names = None

    def _create_model(self):
        """Create base model"""
        if self.model_type == 'random_forest':
            if self.task == 'classification':
                return RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    max_features='sqrt',
                    random_state=self.random_state,
                    n_jobs=-1
                )
            else:
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=self.random_state,
                    n_jobs=-1
                )

        elif self.model_type == 'gradient_boosting':
            if self.task == 'classification':
                return GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=5,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    subsample=0.8,
                    random_state=self.random_state
                )
            else:
                from sklearn.ensemble import GradientBoostingRegressor
                return GradientBoostingRegressor(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=5,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    subsample=0.8,
                    random_state=self.random_state
                )

        elif self.model_type == 'xgb':
            try:
                import xgboost as xgb
                if self.task == 'classification':
                    return xgb.XGBClassifier(
                        n_estimators=150,
                        learning_rate=0.05,
                        max_depth=5,
                        min_child_weight=10,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=self.random_state,
                        n_jobs=-1
                    )
                else:
                    return xgb.XGBRegressor(
                        n_estimators=150,
                        learning_rate=0.05,
                        max_depth=5,
                        min_child_weight=10,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=self.random_state,
                        n_jobs=-1
                    )
            except ImportError:
                logger.warning("XGBoost not available, falling back to RandomForest")
                self.model_type = 'random_forest'
                return self._create_model()

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Fit the model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Dictionary with training metrics
        """
        self.feature_names = list(X_train.columns)

        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None

        # Create and train model
        self.model = self._create_model()

        logger.info(f"Training {self.model_type} ({self.task})...")
        self.model.fit(X_train_scaled, y_train)

        # Calibrate if requested (classification only)
        if self.calibrate and self.task == 'classification' and X_val is not None:
            logger.info("Calibrating probabilities...")
            self.model = CalibratedClassifierCV(
                self.model,
                method='isotonic',
                cv='prefit'
            )
            self.model.fit(X_val_scaled, y_val)

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
        elif hasattr(self.model, 'base_estimator') and hasattr(self.model.base_estimator, 'feature_importances_'):
            # For calibrated models
            self.feature_importance = dict(zip(
                self.feature_names,
                self.model.base_estimator.feature_importances_
            ))

        # Evaluate
        metrics = {}
        if self.task == 'classification':
            metrics['train'] = self._evaluate_classification(X_train_scaled, y_train)
            if X_val is not None:
                metrics['val'] = self._evaluate_classification(X_val_scaled, y_val)
        else:
            metrics['train'] = self._evaluate_regression(X_train_scaled, y_train)
            if X_val is not None:
                metrics['val'] = self._evaluate_regression(X_val_scaled, y_val)

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict labels"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities (classification only)"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def _evaluate_classification(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate classification model"""
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='binary' if len(np.unique(y)) == 2 else 'weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='binary' if len(np.unique(y)) == 2 else 'weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='binary' if len(np.unique(y)) == 2 else 'weighted', zero_division=0)
        }

        # AUC for binary classification
        if len(np.unique(y)) == 2:
            try:
                metrics['auc'] = roc_auc_score(y, y_proba[:, 1])
            except:
                pass

        return metrics

    def _evaluate_regression(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate regression model"""
        y_pred = self.model.predict(X)

        return {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mape': np.mean(np.abs((y - y_pred) / (np.abs(y) + 1e-8))) * 100
        }

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N most important features"""
        if self.feature_importance is None:
            return []

        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:n]


def train_validation_split(
    df: pd.DataFrame,
    val_size: float = 0.2,
    shuffle: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and validation sets
    For time series, maintains temporal order (no shuffle by default)

    Args:
        df: Full dataset with features and labels
        val_size: Fraction for validation
        shuffle: Whether to shuffle (not recommended for time series)

    Returns:
        train_df, val_df
    """
    if shuffle:
        df = df.sample(frac=1, random_state=42)

    split_idx = int(len(df) * (1 - val_size))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    return train_df, val_df


def prepare_features_labels(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = 'label'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and labels from dataframe

    Args:
        df: DataFrame with features and labels
        feature_cols: List of feature column names
        label_col: Label column name

    Returns:
        X (features), y (labels)
    """
    X = df[feature_cols].copy()
    y = df[label_col].copy()

    return X, y
