#!/usr/bin/env python3
"""
ML Trading Signal Generator
A comprehensive, leak-free, deterministic trading signal system
"""
import argparse
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

# Import modules
from src.utils import (
    setup_logging, set_random_seeds, save_json,
    load_watchlist, append_jsonl
)
from src.data_sources import DataFetcher, get_default_universe
from src.features import FeatureEngineer, create_labels
from src.models import MLTradingModel, train_validation_split, prepare_features_labels
from src.backtest import WalkForwardBacktest, calculate_true_accuracy
from src.decision import (
    DecisionPolicy, calculate_quality_score,
    generate_recommendation, generate_rationale, generate_risks
)
from src.news import NewsSentimentAnalyzer
from src.performance import PerformanceTracker
from src.ui import (
    print_header, print_success, print_warning, print_error, print_info,
    print_signal_card, print_scan_table, print_summary,
    print_backtest_summary, print_compact, console
)


VERSION = "2.0.0"


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="ML Trading Signal Generator - Deterministic, leak-free trading signals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single ticker analysis
  python quant_trading_signal.py --ticker MSFT --horizon 3 --compact

  # Scan global stocks with news
  python quant_trading_signal.py --scan --universe global --news-api-key YOUR_KEY --top-n 5

  # Scan Russian stocks
  python quant_trading_signal.py --scan --universe russia --top-n 5

  # With backtesting
  python quant_trading_signal.py --ticker AAPL --backtest --bt-splits 5 --bt-save backtest.json

  # Custom watchlist
  python quant_trading_signal.py --scan --universe custom --watchlist AAPL,MSFT,GOOGL --top-n 3
        """
    )

    # Universe and tickers
    parser.add_argument('--universe', choices=['global', 'russia', 'custom'],
                        default='global', help='Stock universe (default: global)')
    parser.add_argument('--ticker', help='Single ticker to analyze')
    parser.add_argument('--scan', action='store_true', help='Scan multiple tickers')
    parser.add_argument('--watchlist', help='Comma-separated list of tickers (for custom universe)')
    parser.add_argument('--watchlist-file', help='File with tickers (one per line)')

    # Data parameters
    parser.add_argument('--history-days', type=int, default=750,
                        help='Days of history to fetch (default: 750, min: 400)')
    parser.add_argument('--cache-ttl', type=int, default=900,
                        help='Cache TTL in seconds (default: 900)')
    parser.add_argument('--csv', help='CSV file for global data (fallback)')
    parser.add_argument('--csv-ru', help='CSV file for Russia data (fallback)')

    # ML parameters
    parser.add_argument('--horizon', type=int, default=3,
                        help='Prediction horizon in days (default: 3)')
    parser.add_argument('--threshold', type=float, default=0.2,
                        help='Classification threshold in %% (default: 0.2)')
    parser.add_argument('--model', choices=['random_forest', 'gradient_boosting', 'xgb'],
                        default='random_forest', help='Model type (default: random_forest)')
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--calibrate', action='store_true', help='Calibrate probabilities')

    # Backtesting
    parser.add_argument('--backtest', action='store_true', help='Run walk-forward backtest')
    parser.add_argument('--bt-splits', type=int, default=5,
                        help='Number of backtest splits (default: 5)')
    parser.add_argument('--bt-lookback-days', type=int, default=500,
                        help='Backtest lookback days (default: 500)')
    parser.add_argument('--bt-horizon', type=int, help='Backtest horizon (default: same as --horizon)')
    parser.add_argument('--bt-save', help='Save backtest results to JSON file')

    # Costs
    parser.add_argument('--transaction-cost', type=float, default=10,
                        help='Transaction cost in bps (default: 10)')
    parser.add_argument('--slippage', type=float, default=8,
                        help='Slippage in bps (default: 8)')

    # News
    parser.add_argument('--news-provider', choices=['newsapi', 'finnhub', 'alphavantage', 'gdelt'],
                        default='newsapi', help='News provider (default: newsapi)')
    parser.add_argument('--news-api-key', help='News provider API key')
    parser.add_argument('--news-window', type=int, default=3,
                        help='News window in days (default: 3)')
    parser.add_argument('--news-max', type=int, default=25,
                        help='Max news articles (default: 25)')
    parser.add_argument('--news-lang', default='en', help='News language (default: en)')
    parser.add_argument('--alpha-vantage-key', help='Alpha Vantage API key (for data)')

    # Scanning and filtering
    parser.add_argument('--parallel', type=int, default=1,
                        help='Parallel workers (default: 1, not yet implemented)')
    parser.add_argument('--top-n', type=int, help='Show top N signals')
    parser.add_argument('--min-confidence', type=float, default=0.0,
                        help='Minimum confidence filter (default: 0.0)')
    parser.add_argument('--only-actions', help='Filter by actions (e.g., BUY,SELL)')

    # Output
    parser.add_argument('--out-dir', default='signals', help='Output directory (default: signals)')
    parser.add_argument('--save-jsonl', help='Save signals to JSONL file')
    parser.add_argument('--perf-db', help='Performance tracking database (SQLite)')
    parser.add_argument('--perf-jsonl', help='Performance tracking JSONL file')
    parser.add_argument('--compact', action='store_true', help='Compact output')
    parser.add_argument('--output', help='Output JSON file (single ticker)')

    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level (default: INFO)')
    parser.add_argument('--version', action='version', version=f'%(prog)s {VERSION}')

    return parser.parse_args()


def analyze_ticker(
    ticker: str,
    args: argparse.Namespace,
    data_fetcher: DataFetcher,
    decision_policy: DecisionPolicy,
    news_analyzer: Optional[NewsSentimentAnalyzer],
    perf_tracker: Optional[PerformanceTracker]
) -> Optional[Dict[str, Any]]:
    """
    Analyze a single ticker and generate signal

    Returns:
        Result dictionary or None if failed
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Analyzing {ticker}")

    # Fetch data
    df = data_fetcher.fetch_ticker(ticker, args.history_days)
    if df is None or len(df) < 400:
        logger.warning(f"Insufficient data for {ticker}")
        return None

    # Engineer features
    feature_eng = FeatureEngineer(add_calendar_features=False)
    df_features = feature_eng.engineer_features(df)

    if df_features.empty:
        logger.warning(f"Feature engineering failed for {ticker}")
        return None

    # Create labels
    df_labeled = create_labels(
        df_features,
        horizon=args.horizon,
        threshold=args.threshold,
        label_type='binary'
    )

    if len(df_labeled) < 100:
        logger.warning(f"Insufficient labeled data for {ticker}")
        return None

    # Split data
    train_df, val_df = train_validation_split(df_labeled, val_size=0.2, shuffle=False)

    feature_cols = feature_eng.get_feature_columns(df_labeled)
    X_train, y_train = prepare_features_labels(train_df, feature_cols)
    X_val, y_val = prepare_features_labels(val_df, feature_cols)

    # Train model
    model = MLTradingModel(
        model_type=args.model,
        task='classification',
        calibrate=args.calibrate,
        random_state=args.seed
    )

    metrics = model.fit(X_train, y_train, X_val, y_val)
    val_metrics = metrics.get('val', {})

    # Get latest prediction
    latest_row = df_labeled.iloc[[-1]]
    X_latest = latest_row[feature_cols]

    p_up = model.predict_proba(X_latest)[0, 1]  # Probability of up move

    # Get current price and volatility
    current_price = float(latest_row['close'].values[0])
    atr = float(latest_row['atr'].values[0])
    volatility = float(latest_row['realized_vol_20'].values[0])

    # Get news sentiment
    news_result = None
    news_score = None
    if news_analyzer:
        try:
            news_result = news_analyzer.analyze_sentiment(
                ticker,
                days=args.news_window,
                max_articles=args.news_max,
                language=args.news_lang
            )
            news_score = news_result.get('score', 0.0)
        except Exception as e:
            logger.warning(f"News fetch failed for {ticker}: {e}")
            news_result = {'enabled': False, 'score': 0.0}

    # Generate signal
    signal = decision_policy.generate_signal(
        p_up=p_up,
        current_price=current_price,
        atr=atr,
        volatility=volatility,
        mu_estimate=None,  # Let policy estimate
        news_score=news_score
    )

    # Run backtest if requested
    backtest_results = None
    true_accuracy = None
    if args.backtest:
        logger.info(f"Running backtest for {ticker}")
        backtester = WalkForwardBacktest(
            n_splits=args.bt_splits,
            lookback_days=args.bt_lookback_days,
            horizon=args.bt_horizon or args.horizon,
            transaction_cost=args.transaction_cost / 10000,
            slippage=args.slippage / 10000
        )
        backtest_results = backtester.run_backtest(
            df_labeled,
            feature_cols,
            model_type=args.model,
            calibrate=args.calibrate
        )
        true_accuracy = calculate_true_accuracy(backtest_results)

    # Get top features
    top_features = model.get_top_features(n=10)

    # Calculate quality score
    quality = calculate_quality_score(
        signal,
        news_score=news_score,
        true_accuracy=true_accuracy
    )

    # Add additional fields
    signal['recommendation'] = generate_recommendation(signal)
    signal['rationale'] = generate_rationale(signal, top_features, news_result)
    signal['risks'] = generate_risks(signal)

    # Log to performance tracker
    if perf_tracker:
        perf_tracker.log_signal(ticker, signal, quality['quality_score'], news_score)

    # Build result
    result = {
        'ticker': ticker,
        'timestamp': datetime.now().isoformat(),
        'price': current_price,
        'ml_prediction': {
            'p_up': float(p_up),
            'confidence': float(signal['confidence']),
            'expected_return_pct': float(signal['expected_return_pct']),
            'uncertainty_pct': float(signal['uncertainty_pct']),
            'action': signal['action'],
            'recommendation': signal['recommendation']
        },
        'trade_plan': {
            'action': signal['action_raw'],
            'position_size_pct': float(signal['position_size_pct']),
            'stop_loss': float(signal['risk_plan']['stop_loss']) if signal['risk_plan']['stop_loss'] else None,
            'take_profit': float(signal['risk_plan']['take_profit']) if signal['risk_plan']['take_profit'] else None,
            'risk_reward': float(signal['risk_plan']['risk_reward']) if signal['risk_plan']['risk_reward'] else None
        },
        'quality': {
            'score': float(quality['quality_score']),
            'score_100': float(quality['quality_score_100'])
        },
        'model': {
            'type': args.model,
            'validation_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v
                                    for k, v in val_metrics.items()},
            'top_features': [{'name': f[0], 'importance': float(f[1])} for f in top_features[:5]]
        },
        'rationale': signal['rationale'],
        'risks': signal['risks']
    }

    # Add news
    if news_result:
        result['news'] = {
            'enabled': news_result.get('enabled', False),
            'provider': news_result.get('provider'),
            'window': news_result.get('window'),
            'score': float(news_result.get('score', 0)),
            'sources': news_result.get('sources', 0),
            'top_headlines': news_result.get('top_headlines', [])[:3],
            'mu_adj_delta': float(signal.get('news_adjustment', {}).get('mu_delta', 0)),
            'conf_delta': float(signal.get('news_adjustment', {}).get('conf_delta', 0)),
            'explanation': news_result.get('explanation', '')
        }

    # Add backtest results
    if backtest_results:
        result['performance'] = {
            'true_accuracy': float(true_accuracy),
            'trades_count': backtest_results.get('n_trades', 0),
            'lookback_days': args.bt_lookback_days,
            'backtest': {k: float(v) if isinstance(v, (int, float, np.number)) else v
                         for k, v in backtest_results.items()
                         if k not in ['split_metrics']}
        }

    # Store for output
    result['_signal'] = signal  # Internal use
    result['_top_features'] = top_features  # Internal use

    return result


def main():
    """Main entry point"""
    args = parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    # Set random seed
    set_random_seeds(args.seed)

    # Print header
    if not args.compact:
        print_header(f"ML Trading Signal Generator v{VERSION}")

    # Determine tickers
    tickers = []
    if args.ticker:
        tickers = [args.ticker]
    elif args.scan:
        if args.universe == 'custom':
            if args.watchlist:
                tickers = [t.strip() for t in args.watchlist.split(',')]
            elif args.watchlist_file:
                tickers = load_watchlist(args.watchlist_file)
            else:
                print_error("Custom universe requires --watchlist or --watchlist-file")
                sys.exit(1)
        else:
            tickers = get_default_universe(args.universe)
    else:
        print_error("Must specify --ticker or --scan")
        sys.exit(1)

    if not tickers:
        print_error("No tickers to analyze")
        sys.exit(1)

    # Initialize components
    data_fetcher = DataFetcher(
        universe=args.universe,
        alpha_vantage_key=args.alpha_vantage_key,
        csv_path=args.csv,
        csv_ru_path=args.csv_ru,
        cache_ttl=args.cache_ttl
    )

    decision_policy = DecisionPolicy(
        buy_threshold=0.60,
        sell_threshold=0.40,
        min_confidence=0.35,
        transaction_cost=args.transaction_cost / 10000,
        slippage=args.slippage / 10000
    )

    news_analyzer = None
    if args.news_api_key:
        news_analyzer = NewsSentimentAnalyzer(
            provider=args.news_provider,
            api_key=args.news_api_key
        )

    perf_tracker = None
    if args.perf_db or args.perf_jsonl:
        perf_tracker = PerformanceTracker(
            db_path=args.perf_db,
            jsonl_path=args.perf_jsonl
        )

    # Analyze tickers
    results = []
    skipped = 0

    for ticker in tickers:
        try:
            result = analyze_ticker(
                ticker,
                args,
                data_fetcher,
                decision_policy,
                news_analyzer,
                perf_tracker
            )

            if result:
                results.append(result)
                if not args.compact:
                    print_success(f"{ticker} analyzed")
            else:
                skipped += 1
                if not args.compact:
                    print_warning(f"{ticker} skipped (insufficient data)")

        except KeyboardInterrupt:
            print_warning("Interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}", exc_info=True)
            skipped += 1
            if not args.compact:
                print_error(f"{ticker} failed: {e}")

    # Filter results
    if args.min_confidence > 0:
        results = [r for r in results if r['ml_prediction']['confidence'] >= args.min_confidence]

    if args.only_actions:
        allowed_actions = [a.strip().upper() for a in args.only_actions.split(',')]
        results = [r for r in results if r['trade_plan']['action'] in allowed_actions]

    # Sort by quality
    results = sorted(results, key=lambda x: x['quality']['score'], reverse=True)

    # Limit to top N
    if args.top_n:
        results = results[:args.top_n]

    # Output results
    if not results:
        print_warning("No signals generated")
        sys.exit(0)

    # Display
    if args.compact:
        # Compact JSON output
        for result in results:
            # Remove internal fields
            result.pop('_signal', None)
            result.pop('_top_features', None)
        print_compact(results if len(results) > 1 else results[0])
    else:
        if len(results) == 1:
            # Single ticker card
            result = results[0]
            print_signal_card(
                result['ticker'],
                result['_signal'],
                features=result['_top_features'],
                news=result.get('news'),
                backtest=result.get('performance', {}).get('backtest'),
                timestamp=result['timestamp']
            )
        else:
            # Scan table
            scan_results = []
            for result in results:
                scan_results.append({
                    'ticker': result['ticker'],
                    'signal': result['_signal'],
                    'quality_score': result['quality']['score'],
                    'news': result.get('news', {})
                })
            print_scan_table(scan_results)

        # Summary
        summary = {
            'universe': args.universe,
            'analyzed': len(results) + skipped,
            'skipped': skipped,
            'buy_count': sum(1 for r in results if r['trade_plan']['action'] == 'BUY'),
            'sell_count': sum(1 for r in results if r['trade_plan']['action'] == 'SELL'),
            'hold_count': sum(1 for r in results if r['trade_plan']['action'] == 'HOLD'),
            'model_type': args.model,
            'horizon': args.horizon,
            'news_enabled': news_analyzer is not None,
            'news_provider': args.news_provider if news_analyzer else None,
            'backtest_enabled': args.backtest,
            'bt_splits': args.bt_splits if args.backtest else None
        }
        print_summary(summary)

    # Save outputs
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Individual JSON files
    for result in results:
        result_copy = result.copy()
        result_copy.pop('_signal', None)
        result_copy.pop('_top_features', None)

        ticker = result['ticker']
        filepath = Path(args.out_dir) / f"{ticker}.json"
        save_json(result_copy, str(filepath))

    # Summary JSON
    summary_data = {
        'generated_at': datetime.now().isoformat(),
        'parameters': vars(args),
        'summary': summary,
        'top_signals': [
            {
                'ticker': r['ticker'],
                'action': r['trade_plan']['action'],
                'quality_score': r['quality']['score'],
                'expected_return_pct': r['ml_prediction']['expected_return_pct'],
                'confidence': r['ml_prediction']['confidence']
            }
            for r in results[:10]
        ]
    }
    save_json(summary_data, str(Path(args.out_dir) / "summary.json"))

    # JSONL append
    if args.save_jsonl:
        for result in results:
            result_copy = result.copy()
            result_copy.pop('_signal', None)
            result_copy.pop('_top_features', None)
            append_jsonl(result_copy, args.save_jsonl)

    # Single ticker output
    if args.output and len(results) == 1:
        result_copy = results[0].copy()
        result_copy.pop('_signal', None)
        result_copy.pop('_top_features', None)
        save_json(result_copy, args.output)

    if not args.compact:
        print_success(f"Results saved to {args.out_dir}/")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        logging.getLogger(__name__).exception("Fatal error")
        sys.exit(1)
