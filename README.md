# ML Trading Signal Generator

A comprehensive, deterministic, leak-free machine learning trading signal generator with optional news sentiment integration and full accuracy auditing.

**Version:** 2.0.0

## Features

- ✅ **Leak-Free Feature Engineering**: All features at time t use only data ≤ t
- ✅ **Deterministic**: Fixed random seeds ensure reproducible results
- ✅ **Walk-Forward Backtesting**: Purged time-series splits for realistic performance
- ✅ **Multi-Source Data**: Yahoo Finance, MOEX (Russia), Alpha Vantage
- ✅ **ML Models**: Random Forest, Gradient Boosting, XGBoost
- ✅ **News Sentiment**: Optional integration with NewsAPI, Finnhub, Alpha Vantage, GDELT
- ✅ **Performance Tracking**: SQLite/JSONL-based accuracy auditing
- ✅ **Beautiful Terminal UI**: Rich colored tables and cards
- ✅ **Graceful Failures**: Never crashes on missing data or API errors
- ✅ **Backward Compatible**: All existing flags work

## Installation

### Requirements

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional Dependencies

For XGBoost support:
```bash
pip install xgboost
```

## Quick Start

### Single Ticker Analysis

Analyze a single stock with default settings:

```bash
python quant_trading_signal.py --ticker MSFT
```

With compact JSON output:

```bash
python quant_trading_signal.py --ticker MSFT --compact
```

### Scan Global Stocks

Scan default global universe and show top 10 signals:

```bash
python quant_trading_signal.py --scan --universe global --top-n 10
```

### Scan Russian Stocks (MOEX)

```bash
python quant_trading_signal.py --scan --universe russia --top-n 5
```

### With News Sentiment

```bash
python quant_trading_signal.py --scan --universe global --news-api-key YOUR_KEY --top-n 5
```

### With Backtesting

Run walk-forward backtest for comprehensive validation:

```bash
python quant_trading_signal.py --ticker AAPL --horizon 3 --backtest --bt-splits 5 --bt-save backtest.json
```

### Custom Watchlist

```bash
python quant_trading_signal.py --scan --universe custom --watchlist AAPL,MSFT,GOOGL,AMZN --top-n 3
```

Or from file:

```bash
python quant_trading_signal.py --scan --universe custom --watchlist-file my_tickers.txt --top-n 5
```

## Command-Line Reference

### Universe & Tickers

| Flag | Description | Default |
|------|-------------|---------|
| `--universe` | Stock universe: `global`, `russia`, or `custom` | `global` |
| `--ticker` | Single ticker to analyze | - |
| `--scan` | Scan multiple tickers | - |
| `--watchlist` | Comma-separated tickers (for custom) | - |
| `--watchlist-file` | File with tickers (one per line) | - |

### Data Parameters

| Flag | Description | Default |
|------|-------------|---------|
| `--history-days` | Days of history to fetch | 750 |
| `--cache-ttl` | Cache TTL in seconds | 900 |
| `--csv` | CSV file for global data (fallback) | - |
| `--csv-ru` | CSV file for Russia data (fallback) | - |
| `--alpha-vantage-key` | Alpha Vantage API key | - |

### ML Parameters

| Flag | Description | Default |
|------|-------------|---------|
| `--horizon` | Prediction horizon in days | 3 |
| `--threshold` | Classification threshold (%) | 0.2 |
| `--model` | Model: `random_forest`, `gradient_boosting`, `xgb` | `random_forest` |
| `--tune` | Perform hyperparameter tuning | False |
| `--calibrate` | Calibrate probabilities | False |

### Backtesting

| Flag | Description | Default |
|------|-------------|---------|
| `--backtest` | Run walk-forward backtest | False |
| `--bt-splits` | Number of backtest splits | 5 |
| `--bt-lookback-days` | Backtest lookback days | 500 |
| `--bt-horizon` | Backtest horizon (defaults to `--horizon`) | - |
| `--bt-save` | Save backtest results to JSON | - |

### Costs

| Flag | Description | Default |
|------|-------------|---------|
| `--transaction-cost` | Transaction cost in bps | 10 |
| `--slippage` | Slippage in bps | 8 |

### News Sentiment

| Flag | Description | Default |
|------|-------------|---------|
| `--news-provider` | Provider: `newsapi`, `finnhub`, `alphavantage`, `gdelt` | `newsapi` |
| `--news-api-key` | News provider API key | - |
| `--news-window` | News window in days | 3 |
| `--news-max` | Max articles to fetch | 25 |
| `--news-lang` | Language code | `en` |

### Filtering & Output

| Flag | Description | Default |
|------|-------------|---------|
| `--top-n` | Show top N signals | - |
| `--min-confidence` | Minimum confidence filter | 0.0 |
| `--only-actions` | Filter by actions (e.g., `BUY,SELL`) | - |
| `--out-dir` | Output directory | `signals` |
| `--save-jsonl` | Save signals to JSONL file | - |
| `--perf-db` | Performance tracking SQLite DB | - |
| `--perf-jsonl` | Performance tracking JSONL | - |
| `--compact` | Compact JSON output | False |
| `--output` | Output JSON file (single ticker) | - |

### Misc

| Flag | Description | Default |
|------|-------------|---------|
| `--seed` | Random seed for reproducibility | 42 |
| `--log-level` | Logging level | `INFO` |
| `--version` | Show version | - |

## Architecture

### Module Structure

```
trading_signals/
├── quant_trading_signal.py   # Main CLI application
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── src/
│   ├── __init__.py
│   ├── data_sources.py       # Data fetching (Yahoo, MOEX, Alpha Vantage)
│   ├── features.py           # Feature engineering (leak-free)
│   ├── models.py             # ML models (RF, GBM, XGB)
│   ├── backtest.py           # Walk-forward backtesting
│   ├── decision.py           # Signal generation & decision policy
│   ├── news.py               # News sentiment analysis
│   ├── performance.py        # Performance tracking & auditing
│   ├── ui.py                 # Terminal UI (Rich)
│   └── utils.py              # Utilities
└── tests/
    ├── __init__.py
    └── test_basic.py         # Basic tests
```

### Data Flow

```
1. Data Ingestion → 2. Feature Engineering → 3. Label Creation →
4. Model Training → 5. Prediction → 6. Decision Policy →
7. News Adjustment → 8. Signal Output → 9. Performance Tracking
```

### No-Leakage Guarantees

- **Features at time t**: Only use data up to and including time t
- **Labels**: Shifted by +horizon to look ahead correctly
- **Train/Val Split**: Temporal split (no shuffle for time series)
- **Walk-Forward Backtest**: Train on past, predict next period, roll forward

## Output Format

### Single Ticker Output

```json
{
  "ticker": "MSFT",
  "timestamp": "2025-11-02T10:00:00Z",
  "price": 526.12,
  "ml_prediction": {
    "p_up": 0.67,
    "confidence": 0.64,
    "expected_return_pct": 1.0,
    "uncertainty_pct": 1.6,
    "action": "BUY 🔼",
    "recommendation": "Moderate BUY — edge +0.72%, confidence 64%"
  },
  "trade_plan": {
    "action": "BUY",
    "position_size_pct": 8.0,
    "stop_loss": 507.62,
    "take_profit": 553.88,
    "risk_reward": 1.6
  },
  "quality": {
    "score": 0.76,
    "score_100": 76.0
  },
  "model": {
    "type": "random_forest",
    "validation_metrics": {
      "accuracy": 0.68,
      "precision": 0.66,
      "recall": 0.61,
      "f1": 0.63
    },
    "top_features": [
      {"name": "macd_hist", "importance": 0.12},
      {"name": "rsi", "importance": 0.09},
      {"name": "price_to_sma_20", "importance": 0.08}
    ]
  },
  "news": {
    "enabled": true,
    "provider": "newsapi",
    "window": "3d",
    "score": 0.20,
    "sources": 3,
    "top_headlines": [...],
    "mu_adj_delta": 0.20,
    "conf_delta": 0.02,
    "explanation": "Moderately positive sentiment from 8 articles across 3 sources"
  },
  "performance": {
    "true_accuracy": 0.87,
    "trades_count": 124,
    "lookback_days": 180,
    "backtest": {
      "hitrate": 0.62,
      "sharpe": 1.1,
      "sortino": 1.7,
      "max_drawdown_pct": 9.4,
      "avg_win_pct": 2.3,
      "avg_loss_pct": -1.4,
      "cagr_pct": 18.5,
      "turnover": 0.35
    }
  },
  "rationale": "Key factors: macd_hist, rsi, price_to_sma_20; positive risk-reward; supportive news sentiment",
  "risks": "Moderate volatility; watch pullbacks near resistance"
}
```

### Scan Output (Terminal)

```
✓ Analyzed 36 | BUY 8 | SELL 3 | HOLD 25

=====================================================================================
TICKER  | ACTION    |   RET%|  CONF|  NEWS|    PRICE|       TP|       SL| QScore
=====================================================================================
NVDA    | BUY 🔼     |   3.1|  0.81|  +0.6|   132.80|   138.40|   128.90|   0.91
TSLA    | SELL 🔽    |  -2.7|  0.78|  -0.4|   245.20|   238.10|   252.00|   0.86
AAPL    | BUY 🔼     |   2.2|  0.72|  +0.3|   189.50|   194.80|   186.20|   0.79
...
```

## Performance Tracking

Enable performance tracking to build a database of signal outcomes:

```bash
python quant_trading_signal.py --scan --universe global --perf-db signals.db --perf-jsonl perf.jsonl
```

Over time, the system will:
- Log each signal with timestamp, action, confidence, etc.
- Evaluate outcomes after horizon passes
- Calculate true accuracy metrics (hit rate, Sharpe, max DD, edge)
- Use true accuracy in quality scoring for better ranking

## Testing

Run tests:

```bash
pytest tests/ -v
```

Run tests with coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## Acceptance Tests

### Test 1: Single Global Ticker (No News)

```bash
python quant_trading_signal.py --ticker MSFT --horizon 3 --compact
```

**Expected**: JSON output with ml_prediction, trade_plan, model metrics.

### Test 2: Scan Global with News

```bash
python quant_trading_signal.py --scan --universe global --news-api-key YOUR_KEY --top-n 5 --compact
```

**Expected**: Ranked table (or JSON with --compact), per-ticker JSON files, signals/summary.json.

### Test 3: Scan Russian Stocks

```bash
python quant_trading_signal.py --scan --universe russia --top-n 5 --compact
```

**Expected**: MOEX data fetched, signals generated, outputs in local timezone.

### Test 4: Backtest & True Accuracy

```bash
python quant_trading_signal.py --ticker AAPL --horizon 3 --backtest --bt-splits 5 --bt-lookback-days 500 --bt-save backtest.json
```

**Expected**: Walk-forward metrics in backtest.json; true_accuracy in main output.

### Test 5: Performance Persistence

```bash
python quant_trading_signal.py --scan --universe global --top-n 3 --perf-db signals.db
```

**Expected**: signals.db created; subsequent runs update outcomes and confusion matrices.

### Test 6: Graceful Failures

Test with invalid ticker:

```bash
python quant_trading_signal.py --ticker INVALID123
```

**Expected**: Warning logged, no crash.

Test with missing news key (news provider requires key):

```bash
python quant_trading_signal.py --ticker MSFT --news-provider newsapi
```

**Expected**: News skipped, technical/ML signal still generated.

## API Keys (Optional)

### NewsAPI

Get a free key at [newsapi.org](https://newsapi.org/)

```bash
export NEWS_API_KEY="your_key_here"
python quant_trading_signal.py --ticker MSFT --news-api-key $NEWS_API_KEY
```

### Alpha Vantage

Get a free key at [alphavantage.co](https://www.alphavantage.co/support/#api-key)

```bash
python quant_trading_signal.py --ticker MSFT --alpha-vantage-key YOUR_KEY
```

### Finnhub

Get a free key at [finnhub.io](https://finnhub.io/)

```bash
python quant_trading_signal.py --ticker MSFT --news-provider finnhub --news-api-key YOUR_KEY
```

## FAQ

**Q: How do I ensure no look-ahead bias?**

A: The system guarantees no leakage by:
- Features at time t only use data ≤ t
- Labels are properly shifted by horizon
- Walk-forward backtest uses purged splits
- No future data ever used during training or inference

**Q: Is the output deterministic?**

A: Yes, given the same inputs and random seed (--seed), the output will be identical.

**Q: What if a data source is down?**

A: The system gracefully falls back to alternative sources or skips the ticker with a warning. It never crashes.

**Q: Can I use my own CSV data?**

A: Yes, use --csv for global data or --csv-ru for Russian data. Ensure columns: date, open, high, low, close, volume.

**Q: How accurate are the signals?**

A: Use --backtest to see historical performance. Real accuracy varies by market conditions. This is a tool for analysis, not financial advice.

**Q: How do I interpret the quality score?**

A: Quality score (0-1) combines risk-adjusted edge, confidence, historical accuracy, and news sentiment. Higher = better signal quality.

## Troubleshooting

### Import Errors

```bash
pip install -r requirements.txt
```

### XGBoost Not Available

XGBoost is optional. If not installed, the system falls back to RandomForest.

```bash
pip install xgboost
```

### Rate Limiting

If you hit API rate limits, increase --cache-ttl or reduce scan frequency.

### Missing Data

If a ticker has insufficient data (< 400 days), it will be skipped with a warning. Use --log-level DEBUG for details.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Disclaimer

**This software is for educational and research purposes only.**

- Not financial advice
- No warranty or guarantee of accuracy
- Past performance ≠ future results
- Trading involves risk of loss
- Use at your own risk

Always do your own research and consult a licensed financial advisor before making investment decisions.

## Version History

### 2.0.0 (2025-11-02)

- Complete rewrite with ML-based signals
- Walk-forward backtesting
- News sentiment integration
- Performance tracking and accuracy auditing
- Beautiful terminal UI
- Leak-free feature engineering
- Deterministic output
- Graceful error handling
- Backward compatible flags

### 1.0.0

- Initial basic signals (technical only)

---

**Built with ❤️ by senior quant & ML engineers**
