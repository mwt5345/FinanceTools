"""
Historical validation of intraday mean-reversion strategies.

Supports two data feeds:
  - yfinance (default): 1-minute bars, max 7 calendar days
  - alpaca: 1-minute bars, months/years of history (requires API keys)

Supports two strategies:
  - chebyshev (default): Chebyshev inequality z-score
  - ou: Ornstein-Uhlenbeck process signal

Usage:
    python -m finance_tools.backtest.intraday MSFT --days 5
    python -m finance_tools.backtest.intraday AAPL --strategy ou --entry 2.0 --max-threshold 3.0
    python -m finance_tools.backtest.intraday F --feed alpaca --days 60 --strategy ou --figures
"""

import argparse
import math
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

from finance_tools.backtest.engine import Backtester, Portfolio, Action, ActionType, Trade
from finance_tools.broker.data_feed import YFinanceFeed, AlpacaFeed
from finance_tools.strategies.intraday import (IntradayChebyshev, IntradayChebyshevWithCooldown,
                      IntradayOU, IntradayOUWithCooldown)

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def fetch_intraday_data(ticker: str, days: int = 5,
                        feed_type: str = "yfinance") -> pd.DataFrame:
    """Fetch 1-minute bars from the specified data feed.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    days : int
        Number of trading days of data. yfinance caps at ~7 calendar days;
        Alpaca supports months/years.
    feed_type : str
        "yfinance" or "alpaca".

    Returns
    -------
    pd.DataFrame with OHLCV columns.
    """
    # ~390 trading minutes per day
    lookback = days * 390

    if feed_type == "alpaca":
        feed = AlpacaFeed(ticker)
    else:
        feed = YFinanceFeed(ticker)

    hist = feed.history(lookback_minutes=lookback)
    return hist


def compute_intraday_metrics(result, interval_seconds: int = 60) -> dict:
    """Compute annualized metrics for intraday backtest.

    Aggregates 1-minute portfolio values to end-of-day values, then
    computes Sharpe / vol / return using daily returns with sqrt(252)
    annualization.  This makes the metrics directly comparable to the
    daily-bar backtests elsewhere in the project.

    Parameters
    ----------
    result : BacktestResult
    interval_seconds : int
        Bar interval in seconds (default 60 for 1m bars).

    Returns
    -------
    dict with ann_return, ann_vol, sharpe, max_drawdown, n_bars,
    n_trades, n_days, total_return.
    """
    values = result.daily_values
    n_bars = len(values)

    if n_bars < 2:
        return {
            "ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0,
            "max_drawdown": 0.0, "n_bars": n_bars, "n_trades": 0,
            "n_days": 0, "total_return": 0.0,
        }

    total_return = result.final_value / result.initial_cash - 1

    # Aggregate to end-of-day values for daily-scale metrics.
    # The index is a DatetimeIndex from the 1m bars — group by date
    # and take the last value each day.
    idx = values.index
    if hasattr(idx[0], "date"):
        dates = pd.Series(values.values, index=[d.date() for d in idx])
    else:
        dates = pd.Series(values.values, index=idx)
    eod_values = dates.groupby(dates.index).last()
    n_days = len(eod_values)

    if n_days >= 2:
        daily_returns = eod_values.pct_change().dropna()
        ann_vol = float(daily_returns.std() * np.sqrt(252))
        ann_return = (1 + total_return) ** (252 / n_days) - 1
    else:
        # Only 1 trading day — fall back to bar-level annualization
        trading_seconds_per_year = 252 * 6.5 * 3600
        bars_per_year = trading_seconds_per_year / interval_seconds
        ann_return = (1 + total_return) ** (bars_per_year / n_bars) - 1
        returns = values.pct_change().dropna()
        ann_vol = float(returns.std() * np.sqrt(bars_per_year))

    sharpe = 0.0
    if ann_vol > 0:
        sharpe = ann_return / ann_vol

    # Max drawdown (bar-level for accuracy)
    cummax = values.cummax()
    drawdown = (values - cummax) / cummax
    max_dd = float(drawdown.min())

    n_trades = len([t for t in result.trades if t.action not in ("hold", "dividend")])

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_bars": n_bars,
        "n_trades": n_trades,
        "n_days": n_days,
        "total_return": total_return,
    }


def run_backtest(ticker: str, days: int = 5, cash: float = 10_000.0,
                 k: float = 2.0, window: int = 30, cooldown: int = 6,
                 strategy_type: str = "chebyshev",
                 entry: float = 2.0, max_threshold: float = 3.0,
                 feed_type: str = "yfinance",
                 verbose: bool = True):
    """Run an intraday backtest and print results.

    Parameters
    ----------
    strategy_type : str
        "chebyshev" or "ou".
    entry : float
        OU entry threshold (ignored for chebyshev).
    max_threshold : float
        OU max threshold for full position sizing (ignored for chebyshev).
    feed_type : str
        "yfinance" or "alpaca".

    Returns (BacktestResult, metrics_dict, hist_DataFrame, strategy).
    """
    if verbose:
        feed_label = "Alpaca" if feed_type == "alpaca" else "yfinance"
        print(f"\n  {BOLD}{CYAN}Intraday Backtest: {ticker}{RESET}")
        print(f"  Fetching {days} days of 1-minute data via {feed_label}...")

    hist = fetch_intraday_data(ticker, days, feed_type=feed_type)

    if len(hist) == 0:
        if verbose:
            print(f"  {RED}No data available for {ticker}.{RESET}")
        return None, None, None, None

    if verbose:
        print(f"  Got {len(hist)} bars from {hist.index[0]} to {hist.index[-1]}")

    # Construct strategy
    if strategy_type == "ou":
        strategy = IntradayOUWithCooldown(
            window=window, entry_threshold=entry,
            max_threshold=max_threshold, cooldown_ticks=cooldown)
    else:
        strategy = IntradayChebyshevWithCooldown(
            window=window, k_threshold=k, cooldown_ticks=cooldown)

    bt = Backtester(hist, strategy, initial_cash=cash, cash_reserve_pct=0.05)
    result = bt.run()

    # Compute intraday-aware metrics (1m = 60s bars)
    metrics = compute_intraday_metrics(result, interval_seconds=60)

    if verbose:
        print(f"\n  {BOLD}Results:{RESET}")
        print(f"  Strategy:       {strategy.name}")
        print(f"  Initial cash:   ${cash:,.0f}")
        print(f"  Final value:    ${result.final_value:,.2f}")
        ret_color = GREEN if metrics["total_return"] >= 0 else RED
        print(f"  Total return:   {ret_color}{metrics['total_return']:+.2%}{RESET}")
        print(f"  Ann. return:    {metrics['ann_return']:+.2%}")
        print(f"  Ann. vol:       {metrics['ann_vol']:.2%}")
        print(f"  Sharpe:         {metrics['sharpe']:.2f}")
        print(f"  Max drawdown:   {metrics['max_drawdown']:.2%}")
        print(f"  Trades:         {metrics['n_trades']}")
        print(f"  Trading days:   {metrics['n_days']}")
        print(f"  Bars:           {metrics['n_bars']}")

    return result, metrics, hist, strategy


def _compute_signal_series(hist: pd.DataFrame, strategy) -> np.ndarray:
    """Replay strategy.compute_z() across the history to build a signal series."""
    n = len(hist)
    signals = np.full(n, np.nan)
    for i in range(1, n + 1):
        sub = hist.iloc[:i]
        z = strategy.compute_z(sub)
        if z is not None:
            signals[i - 1] = z
    return signals


def generate_backtest_figures(ticker: str, result, metrics: dict,
                              hist: pd.DataFrame = None,
                              strategy=None) -> None:
    """Generate 4 publication-quality backtest figures.

    1. Price with buy/sell markers
    2. Signal (z-score or OU signal s) with threshold bands
    3. Portfolio value vs buy-and-hold
    4. Cumulative P&L with drawdown shading
    """
    from finance_tools.utils.plotting import (
        setup_style, savefig, PALETTE, FIGSIZE,
    )
    import matplotlib.pyplot as plt

    setup_style()

    # Extract strategy info (needed for fig_dir)
    strat_name = strategy.name if strategy else "Strategy"
    is_ou = isinstance(strategy, (IntradayOU, IntradayOUWithCooldown))

    strat_tag = "ou" if is_ou else "chebyshev"
    fig_dir = os.path.join(SCRIPT_DIR, "figures", f"backtest_{strat_tag}")
    os.makedirs(fig_dir, exist_ok=True)

    values = result.daily_values
    bar_idx = np.arange(len(values))

    # Prices from the original history (aligned to result index)
    if hist is not None:
        prices = hist["Close"].values[:len(values)]
    else:
        # Fallback: approximate price from portfolio value
        prices = values.values

    # Trade locations mapped to bar indices
    trade_dates = {d: i for i, d in enumerate(values.index)}
    buy_bars, buy_prices = [], []
    sell_bars, sell_prices = [], []
    for trade in result.trades:
        if trade.action in ("hold", "dividend"):
            continue
        bar_i = trade_dates.get(trade.date)
        if bar_i is None:
            # Find closest bar
            diffs = np.abs((values.index - trade.date).total_seconds())
            bar_i = int(np.argmin(diffs))
        if trade.action == "buy":
            buy_bars.append(bar_i)
            buy_prices.append(trade.price)
        elif trade.action == "sell":
            sell_bars.append(bar_i)
            sell_prices.append(trade.price)

    # ── Figure 1: Price with buy/sell markers ──
    fig, ax = plt.subplots(figsize=FIGSIZE["double"])
    ax.plot(bar_idx, prices, color=PALETTE["blue"], lw=0.8, label="Price")
    if buy_bars:
        ax.scatter(buy_bars, buy_prices, color=PALETTE["green"],
                   marker="^", s=40, zorder=5, label="Buy")
    if sell_bars:
        ax.scatter(sell_bars, sell_prices, color=PALETTE["red"],
                   marker="v", s=40, zorder=5, label="Sell")
    ax.set_xlabel("Bar (1-min)")
    ax.set_ylabel(f"{ticker} Price (\\$)")
    ax.set_title(f"{ticker} --- {strat_name}")
    ax.legend(loc="best")
    savefig(fig, os.path.join(fig_dir, "backtest_price.png"))

    # ── Figure 2: Signal with threshold bands ──
    if strategy is not None and hist is not None:
        # Replay the strategy's compute_z across the history
        # Use a fresh (non-cooldown) strategy for clean signal
        if is_ou:
            sig_strategy = IntradayOU(
                window=strategy.window,
                entry_threshold=strategy.entry_threshold,
                max_threshold=strategy.max_threshold,
                dt=strategy.dt,
            )
            threshold = strategy.entry_threshold
            sig_label = "OU Signal ($s$)"
            thresh_label = "entry"
        else:
            sig_strategy = IntradayChebyshev(
                window=strategy.window,
                k_threshold=strategy.k_threshold,
            )
            threshold = strategy.k_threshold
            sig_label = "z-score"
            thresh_label = "k"

        signals = _compute_signal_series(hist.iloc[:len(values)], sig_strategy)

        fig, ax = plt.subplots(figsize=FIGSIZE["double"])
        ax.plot(bar_idx, signals, color=PALETTE["blue"], lw=0.8,
                label=sig_label)
        ax.axhline(threshold, color=PALETTE["red"], ls="--", lw=1.0,
                   label=f"$+{thresh_label} = {threshold}$")
        ax.axhline(-threshold, color=PALETTE["green"], ls="--", lw=1.0,
                   label=f"$-{thresh_label} = {threshold}$")
        ax.axhline(0, color=PALETTE["grey"], ls=":", lw=0.8, alpha=0.5)
        ax.fill_between(bar_idx, -threshold, threshold,
                        color=PALETTE["grey"], alpha=0.08)
        ax.set_xlabel("Bar (1-min)")
        ax.set_ylabel(sig_label)
        ax.set_title(f"{ticker} {sig_label}")
        ax.legend(loc="best")
        savefig(fig, os.path.join(fig_dir, "backtest_signal.png"))

    # ── Figure 3: Portfolio value vs buy-and-hold ──
    buy_and_hold = result.initial_cash / prices[0] * prices

    fig, ax = plt.subplots(figsize=FIGSIZE["double"])
    ax.plot(bar_idx, values.values, color=PALETTE["blue"], lw=1.0,
            label=strat_name)
    ax.plot(bar_idx, buy_and_hold, color=PALETTE["grey"], lw=1.0,
            ls="--", label="Buy \\& Hold")
    ax.set_xlabel("Bar (1-min)")
    ax.set_ylabel("Portfolio Value (\\$)")
    ax.set_title(f"{ticker} --- Strategy vs Buy \\& Hold")
    ax.legend(loc="best")

    # Annotation box with key metrics
    ret_str = f"{metrics['total_return']:+.2%}"
    sharpe_str = f"{metrics['sharpe']:.2f}"
    trades_str = f"{metrics['n_trades']}"
    textstr = f"Return: {ret_str}\nSharpe: {sharpe_str}\nTrades: {trades_str}"
    props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="grey")
    ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=7,
            verticalalignment="top", bbox=props)
    savefig(fig, os.path.join(fig_dir, "backtest_comparison.png"))

    # ── Figure 4: Cumulative P&L ──
    pnl = values.values - result.initial_cash

    fig, ax = plt.subplots(figsize=FIGSIZE["double"])
    color = PALETTE["green"] if pnl[-1] >= 0 else PALETTE["red"]
    ax.plot(bar_idx, pnl, color=color, lw=1.0)
    ax.axhline(0, color=PALETTE["grey"], ls=":", lw=0.8, alpha=0.5)
    ax.fill_between(bar_idx, 0, pnl, where=(pnl >= 0),
                    color=PALETTE["green"], alpha=0.15)
    ax.fill_between(bar_idx, 0, pnl, where=(pnl < 0),
                    color=PALETTE["red"], alpha=0.15)
    ax.set_xlabel("Bar (1-min)")
    ax.set_ylabel("Cumulative P\\&L (\\$)")
    ax.set_title(f"{ticker} --- {strat_name} P\\&L")
    savefig(fig, os.path.join(fig_dir, "backtest_pnl.png"))

    print(f"\n  {GREEN}Generated 4 figures in {fig_dir}/{RESET}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Intraday Mean-Reversion Backtest")
    parser.add_argument("ticker", nargs="?", default="F",
                        help="Stock ticker (default: F)")
    parser.add_argument("--days", type=int, default=5,
                        help="Trading days of 1m data (yfinance: max 7; "
                        "alpaca: unlimited) (default: 5)")
    parser.add_argument("--feed", choices=["yfinance", "alpaca"],
                        default="yfinance",
                        help="Data feed (default: yfinance)")
    parser.add_argument("--cash", type=float, default=10_000.0,
                        help="Starting cash (default: $10,000)")
    parser.add_argument("--strategy", choices=["chebyshev", "ou"],
                        default="chebyshev",
                        help="Strategy type (default: chebyshev)")
    parser.add_argument("--k", type=float, default=1.5,
                        help="Chebyshev k threshold (default: 1.5)")
    parser.add_argument("--entry", type=float, default=2.0,
                        help="OU entry threshold (default: 2.0)")
    parser.add_argument("--max-threshold", type=float, default=3.0,
                        help="OU max threshold for full sizing (default: 3.0)")
    parser.add_argument("--window", type=int, default=30,
                        help="Rolling window in ticks (default: 30)")
    parser.add_argument("--cooldown", type=int, default=6,
                        help="Cooldown ticks (default: 6)")
    parser.add_argument("--figures", action="store_true",
                        help="Generate plots")
    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()
    ticker = args.ticker.upper()
    days = args.days
    if args.feed == "yfinance":
        days = min(days, 7)  # yfinance 1m limit

    result, metrics, hist, strategy = run_backtest(
        ticker, days=days, cash=args.cash, k=args.k,
        window=args.window, cooldown=args.cooldown,
        strategy_type=args.strategy,
        entry=args.entry, max_threshold=args.max_threshold,
        feed_type=args.feed)

    if result is not None and args.figures:
        generate_backtest_figures(ticker, result, metrics,
                                 hist=hist, strategy=strategy)


if __name__ == "__main__":
    main()
