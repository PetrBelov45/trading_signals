"""
Decision Policy Module - Convert ML predictions to trading signals
Includes position sizing, risk management, and signal quality scoring
"""
import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class DecisionPolicy:
    """
    Convert ML predictions to actionable trading signals
    """

    def __init__(
        self,
        buy_threshold: float = 0.60,
        sell_threshold: float = 0.40,
        min_confidence: float = 0.35,
        transaction_cost: float = 0.001,  # 10 bps
        slippage: float = 0.0008,  # 8 bps
        min_position_pct: float = 5.0,
        max_position_pct: float = 15.0,
        stop_atr_multiplier: float = 2.0,
        take_atr_multiplier: float = 3.0
    ):
        """
        Initialize decision policy

        Args:
            buy_threshold: Probability threshold for BUY
            sell_threshold: Probability threshold for SELL
            min_confidence: Minimum confidence to act
            transaction_cost: Transaction cost (as decimal)
            slippage: Slippage (as decimal)
            min_position_pct: Minimum position size (%)
            max_position_pct: Maximum position size (%)
            stop_atr_multiplier: Stop loss as multiple of ATR
            take_atr_multiplier: Take profit as multiple of ATR
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_confidence = min_confidence
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.total_costs = transaction_cost + slippage
        self.min_position_pct = min_position_pct
        self.max_position_pct = max_position_pct
        self.stop_atr_multiplier = stop_atr_multiplier
        self.take_atr_multiplier = take_atr_multiplier

    def generate_signal(
        self,
        p_up: float,
        current_price: float,
        atr: float,
        volatility: float,
        mu_estimate: Optional[float] = None,
        news_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate trading signal from ML prediction

        Args:
            p_up: Probability of upward move (0-1)
            current_price: Current price
            atr: Average True Range
            volatility: Realized volatility (annualized)
            mu_estimate: Expected return estimate (optional, in %)
            news_score: News sentiment score (optional, -1 to 1)

        Returns:
            Dictionary with signal and risk plan
        """
        # Calculate base confidence
        confidence = abs(p_up - 0.5) * 2  # Scale to 0-1

        # Estimate expected return if not provided
        if mu_estimate is None:
            # Simple mapping from probability to expected return
            mu_estimate = (p_up - 0.5) * 10  # Scale to roughly -5% to +5%

        # Estimate uncertainty (sigma) from volatility
        sigma = volatility / np.sqrt(252) * 100  # Daily vol in %

        # Apply news adjustment if available
        mu_adj = mu_estimate
        conf_adj = confidence
        news_adjustment = None

        if news_score is not None and abs(news_score) >= 0.25:
            # Tilt expected return
            news_tilt = 1.0 * news_score  # Up to ±1% adjustment

            # Reduce tilt if volatility is high
            if sigma > 3.0:
                news_tilt *= 0.5

            mu_adj = mu_estimate + news_tilt
            conf_adj = np.clip(confidence + 0.08 * news_score, 0, 1)

            news_adjustment = {
                'mu_delta': news_tilt,
                'conf_delta': conf_adj - confidence
            }

        # Net edge after costs
        net_edge = mu_adj - (self.total_costs * 100)  # Convert costs to %

        # Decision logic
        action = "HOLD"
        emoji = "⏸️"

        if p_up >= self.buy_threshold and conf_adj >= self.min_confidence and net_edge > 0:
            action = "BUY"
            emoji = "🔼"
        elif p_up <= self.sell_threshold and conf_adj >= self.min_confidence and net_edge < 0:
            action = "SELL"
            emoji = "🔽"

        # Position sizing
        position_size = self._calculate_position_size(conf_adj)

        # Risk plan
        risk_plan = self._calculate_risk_plan(
            action, current_price, atr, position_size
        )

        # Build signal
        signal = {
            'action': f"{action} {emoji}",
            'action_raw': action,
            'p_up': p_up,
            'confidence': conf_adj,
            'expected_return_pct': mu_adj,
            'uncertainty_pct': sigma,
            'net_edge_pct': net_edge,
            'current_price': current_price,
            'position_size_pct': position_size,
            'risk_plan': risk_plan,
            'news_adjustment': news_adjustment
        }

        return signal

    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on confidence"""
        return self.min_position_pct + (self.max_position_pct - self.min_position_pct) * confidence

    def _calculate_risk_plan(
        self,
        action: str,
        price: float,
        atr: float,
        position_size: float
    ) -> Dict[str, Any]:
        """Calculate stop loss and take profit levels"""
        if action == "HOLD":
            return {
                'stop_loss': None,
                'take_profit': None,
                'risk_reward': None
            }

        if action == "BUY":
            stop_loss = price - (self.stop_atr_multiplier * atr)
            take_profit = price + (self.take_atr_multiplier * atr)
        else:  # SELL
            stop_loss = price + (self.stop_atr_multiplier * atr)
            take_profit = price - (self.take_atr_multiplier * atr)

        risk = abs(price - stop_loss) / price
        reward = abs(take_profit - price) / price
        risk_reward = reward / risk if risk > 0 else 0

        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward': risk_reward,
            'risk_pct': risk * 100,
            'reward_pct': reward * 100
        }


def calculate_quality_score(
    signal: Dict[str, Any],
    news_score: Optional[float] = None,
    true_accuracy: Optional[float] = None
) -> Dict[str, float]:
    """
    Calculate composite quality score for signal ranking

    Args:
        signal: Signal dictionary from generate_signal
        news_score: News sentiment score (optional)
        true_accuracy: Historical true accuracy from backtest (optional)

    Returns:
        Dictionary with quality_score (0-1) and quality_score_100 (0-100)
    """
    # Components
    confidence = signal['confidence']
    net_edge = signal['net_edge_pct']
    uncertainty = signal['uncertainty_pct']

    # Risk-adjusted edge (normalize)
    risk_adj_edge = net_edge / (uncertainty + 1e-8)
    edge_norm = np.clip((risk_adj_edge + 2) / 4, 0, 1)

    # News score (normalize)
    news_norm = (abs(news_score) if news_score is not None else 0) / 1.0

    # True accuracy (if available)
    true_acc_norm = true_accuracy if true_accuracy is not None else 0.5

    # Combine with weights
    weights = {
        'edge': 0.3,
        'confidence': 0.3,
        'true_accuracy': 0.25,
        'news': 0.15
    }

    quality_score = (
        weights['edge'] * edge_norm +
        weights['confidence'] * confidence +
        weights['true_accuracy'] * true_acc_norm +
        weights['news'] * news_norm
    )

    quality_score = np.clip(quality_score, 0, 1)

    return {
        'quality_score': quality_score,
        'quality_score_100': quality_score * 100
    }


def generate_recommendation(signal: Dict[str, Any]) -> str:
    """
    Generate human-readable recommendation

    Args:
        signal: Signal dictionary

    Returns:
        Recommendation string
    """
    action = signal['action_raw']
    confidence = signal['confidence']
    net_edge = signal['net_edge_pct']

    if action == "HOLD":
        return "HOLD — no clear edge or insufficient confidence"

    # Classify strength
    if confidence >= 0.75 and abs(net_edge) >= 1.5:
        strength = "Strong"
    elif confidence >= 0.55 and abs(net_edge) >= 0.8:
        strength = "Moderate"
    else:
        strength = "Weak"

    return f"{strength} {action} — edge {net_edge:+.2f}%, confidence {confidence:.0%}"


def generate_rationale(
    signal: Dict[str, Any],
    top_features: list,
    news_sentiment: Optional[Dict] = None
) -> str:
    """
    Generate rationale for the signal

    Args:
        signal: Signal dictionary
        top_features: List of (feature_name, importance) tuples
        news_sentiment: News sentiment dictionary (optional)

    Returns:
        Rationale string
    """
    parts = []

    # Top features
    if top_features:
        feature_names = ", ".join([f[0] for f in top_features[:3]])
        parts.append(f"Key factors: {feature_names}")

    # Edge
    net_edge = signal['net_edge_pct']
    if net_edge > 1.0:
        parts.append("positive risk-reward")
    elif net_edge < -1.0:
        parts.append("negative outlook")

    # Volatility
    uncertainty = signal['uncertainty_pct']
    if uncertainty > 3.0:
        parts.append("elevated volatility")
    elif uncertainty < 1.5:
        parts.append("low volatility")

    # News
    if news_sentiment and abs(news_sentiment.get('score', 0)) >= 0.25:
        if news_sentiment['score'] > 0:
            parts.append("supportive news sentiment")
        else:
            parts.append("negative news sentiment")

    if not parts:
        return "Algorithmic signal based on technical indicators"

    return "; ".join(parts).capitalize()


def generate_risks(signal: Dict[str, Any]) -> str:
    """
    Generate risk warnings

    Args:
        signal: Signal dictionary

    Returns:
        Risk string
    """
    risks = []

    uncertainty = signal['uncertainty_pct']
    if uncertainty > 3.0:
        risks.append("high volatility")
    elif uncertainty > 2.0:
        risks.append("moderate volatility")

    confidence = signal['confidence']
    if confidence < 0.5:
        risks.append("low confidence")

    risk_plan = signal.get('risk_plan', {})
    rr = risk_plan.get('risk_reward', 0)
    if rr < 1.0:
        risks.append("unfavorable risk-reward")

    if not risks:
        return "Standard market risks apply"

    return "; ".join(risks).capitalize()
