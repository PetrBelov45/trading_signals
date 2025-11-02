"""
Terminal UI Module - Beautiful terminal output with colors and formatting
"""
import logging
from typing import Dict, Any, List, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from rich.text import Text


logger = logging.getLogger(__name__)
console = Console()


def print_header(text: str):
    """Print styled header"""
    console.print(f"\n[bold cyan]{text}[/bold cyan]\n")


def print_success(text: str):
    """Print success message"""
    console.print(f"[green]✓[/green] {text}")


def print_warning(text: str):
    """Print warning message"""
    console.print(f"[yellow]⚠[/yellow] {text}")


def print_error(text: str):
    """Print error message"""
    console.print(f"[red]✗[/red] {text}")


def print_info(text: str):
    """Print info message"""
    console.print(f"[blue]ℹ[/blue] {text}")


def print_signal_card(
    ticker: str,
    signal: Dict[str, Any],
    features: Optional[List[tuple]] = None,
    news: Optional[Dict[str, Any]] = None,
    backtest: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None
):
    """
    Print detailed signal card for single ticker

    Args:
        ticker: Stock ticker
        signal: Signal dictionary
        features: Top features [(name, importance), ...]
        news: News sentiment dictionary
        backtest: Backtest results dictionary
        timestamp: Timestamp string
    """
    action = signal['action_raw']
    emoji = "🔼" if action == "BUY" else "🔽" if action == "SELL" else "⏸️"

    # Color based on action
    if action == "BUY":
        color = "green"
    elif action == "SELL":
        color = "red"
    else:
        color = "yellow"

    # Header
    header = f"[bold {color}]{ticker} — {action} {emoji}[/bold {color}]     "
    header += f"Prob↑ {signal['p_up']*100:.0f}%  |  "

    # Confidence bar
    conf = signal['confidence']
    bar = get_progress_bar(conf, width=10)
    header += f"Confidence {bar} {conf:.2f}"

    # Build card content
    lines = [
        "─" * 80,
        f"Price: ${signal['current_price']:.2f} | μ: {signal['expected_return_pct']:+.1f}% | "
        f"σ: {signal['uncertainty_pct']:.1f}% | Net Edge: {signal['net_edge_pct']:+.2f}% | "
        f"RR: {signal['risk_plan'].get('risk_reward', 0):.1f}",
    ]

    # Risk plan
    risk_plan = signal['risk_plan']
    if risk_plan['take_profit'] and risk_plan['stop_loss']:
        lines.append(
            f"TP: ${risk_plan['take_profit']:.2f} | "
            f"SL: ${risk_plan['stop_loss']:.2f} | "
            f"Pos Size: {signal['position_size_pct']:.0f}%"
        )

    # Top features
    if features:
        feature_names = ", ".join([f[0] for f in features[:5]])
        lines.append(f"Top Features: {feature_names}")

    # Model performance
    if backtest:
        bt = backtest
        lines.append(
            f"Model (Val): Acc {bt.get('accuracy', 0)*100:.0f}% | "
            f"Precision {bt.get('precision', 0)*100:.0f}% | "
            f"Recall {bt.get('recall', 0)*100:.0f}% | "
            f"F1 {bt.get('f1', 0)*100:.0f}% (walk-forward)"
        )

    # News
    if news and news.get('enabled'):
        news_sign = "+" if news['score'] > 0 else ""
        lines.append(
            f"News: {news_sign}{news['score']:.2f} ({news['sources']} sources) → "
            f"μ {signal.get('news_adjustment', {}).get('mu_delta', 0):+.2f}pp, "
            f"conf {signal.get('news_adjustment', {}).get('conf_delta', 0):+.2f}"
        )
        lines.append(f"     {news.get('explanation', '')}")

    # Rationale
    if 'rationale' in signal:
        lines.append(f"Rationale: {signal['rationale']}")

    # Risks
    if 'risks' in signal:
        lines.append(f"Risks: {signal['risks']}")

    # Timestamp
    if timestamp:
        lines.append(f"Timestamp: {timestamp}")

    # Print card
    console.print("=" * 80)
    console.print(f"  {header}")
    console.print("=" * 80)
    for line in lines:
        console.print(f"{line}")
    console.print("=" * 80)
    console.print()


def print_scan_table(
    results: List[Dict[str, Any]],
    sort_by: str = "quality_score"
):
    """
    Print ranked table of signals from scan

    Args:
        results: List of result dictionaries with ticker, signal, quality_score, etc.
        sort_by: Field to sort by
    """
    if not results:
        print_warning("No signals to display")
        return

    # Sort results
    results = sorted(results, key=lambda x: x.get(sort_by, 0), reverse=True)

    # Create table
    table = Table(
        title="[bold]Trading Signals Ranked by Quality[/bold]",
        box=box.DOUBLE_EDGE,
        show_header=True,
        header_style="bold cyan"
    )

    table.add_column("TICKER", style="bold", width=8)
    table.add_column("ACTION", width=12)
    table.add_column("RET%", justify="right", width=7)
    table.add_column("CONF", justify="right", width=6)
    table.add_column("NEWS", justify="right", width=6)
    table.add_column("PRICE", justify="right", width=10)
    table.add_column("TP", justify="right", width=10)
    table.add_column("SL", justify="right", width=10)
    table.add_column("QScore", justify="right", width=7)

    # Counters for summary
    buy_count = 0
    sell_count = 0
    hold_count = 0

    for result in results:
        ticker = result['ticker']
        signal = result['signal']
        quality = result.get('quality_score', 0)
        news_score = result.get('news', {}).get('score', 0)

        action = signal['action_raw']
        if action == "BUY":
            action_str = "[green]BUY 🔼[/green]"
            buy_count += 1
        elif action == "SELL":
            action_str = "[red]SELL 🔽[/red]"
            sell_count += 1
        else:
            action_str = "[yellow]HOLD ⏸️[/yellow]"
            hold_count += 1

        # Format values
        ret_pct = f"{signal['expected_return_pct']:+.1f}"
        conf = f"{signal['confidence']:.2f}"
        news_str = f"{news_score:+.1f}" if news_score != 0 else "—"
        price = f"${signal['current_price']:.2f}"

        risk_plan = signal['risk_plan']
        tp = f"${risk_plan.get('take_profit', 0):.2f}" if risk_plan.get('take_profit') else "—"
        sl = f"${risk_plan.get('stop_loss', 0):.2f}" if risk_plan.get('stop_loss') else "—"

        q_score = f"{quality:.2f}"

        table.add_row(
            ticker,
            action_str,
            ret_pct,
            conf,
            news_str,
            price,
            tp,
            sl,
            q_score
        )

    # Print summary header
    summary = f"✓ Analyzed {len(results)} | BUY {buy_count} | SELL {sell_count} | HOLD {hold_count}"
    console.print(f"\n[bold cyan]{summary}[/bold cyan]\n")

    # Print table
    console.print(table)
    console.print()


def get_progress_bar(value: float, width: int = 10, filled: str = "█", empty: str = "░") -> str:
    """
    Generate progress bar

    Args:
        value: Value between 0 and 1
        width: Width of bar
        filled: Character for filled portion
        empty: Character for empty portion

    Returns:
        Progress bar string
    """
    filled_width = int(value * width)
    return filled * filled_width + empty * (width - filled_width)


def print_summary(summary: Dict[str, Any]):
    """
    Print summary statistics

    Args:
        summary: Summary dictionary
    """
    lines = [
        f"Universe: {summary.get('universe', 'N/A')}",
        f"Analyzed: {summary.get('analyzed', 0)} tickers",
        f"Skipped: {summary.get('skipped', 0)} tickers",
        f"Signals: BUY {summary.get('buy_count', 0)}, SELL {summary.get('sell_count', 0)}, HOLD {summary.get('hold_count', 0)}",
        f"Model: {summary.get('model_type', 'N/A')}",
        f"Horizon: {summary.get('horizon', 'N/A')} days",
    ]

    if summary.get('news_enabled'):
        lines.append(f"News: {summary.get('news_provider', 'N/A')}")

    if summary.get('backtest_enabled'):
        lines.append(f"Backtest: {summary.get('bt_splits', 'N/A')} splits")

    panel = Panel(
        "\n".join(lines),
        title="[bold]Run Summary[/bold]",
        border_style="cyan"
    )

    console.print(panel)
    console.print()


def print_progress_spinner(text: str):
    """Show progress spinner (for long operations)"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        progress.add_task(description=text, total=None)


def format_backtest_results(results: Dict[str, Any]) -> str:
    """
    Format backtest results for display

    Args:
        results: Backtest results dictionary

    Returns:
        Formatted string
    """
    lines = [
        f"Hit Rate: {results.get('hitrate', 0)*100:.1f}%",
        f"Sharpe: {results.get('sharpe', 0):.2f}",
        f"Sortino: {results.get('sortino', 0):.2f}",
        f"Max Drawdown: {results.get('max_drawdown_pct', 0):.1f}%",
        f"Avg Win: {results.get('avg_win_pct', 0):.2f}%",
        f"Avg Loss: {results.get('avg_loss_pct', 0):.2f}%",
        f"CAGR: {results.get('cagr_pct', 0):.1f}%",
        f"Trades: {results.get('n_trades', 0)}",
        f"Turnover: {results.get('turnover', 0):.2f}"
    ]

    return " | ".join(lines)


def print_backtest_summary(results: Dict[str, Any]):
    """
    Print backtest summary in a panel

    Args:
        results: Backtest results dictionary
    """
    content = format_backtest_results(results)

    panel = Panel(
        content,
        title="[bold]Backtest Results[/bold]",
        border_style="green"
    )

    console.print(panel)
    console.print()


def print_compact(data: Dict[str, Any]):
    """
    Print compact JSON output

    Args:
        data: Data dictionary
    """
    import json
    console.print(json.dumps(data, indent=2, default=str))
