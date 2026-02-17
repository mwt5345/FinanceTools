"""
Intraday Trader -- Live polling and streaming CLI.

Polls a single stock, applies a mean-reversion strategy (Chebyshev or
Ornstein-Uhlenbeck), and trades with whole shares.

Supports two brokers:
  - local (default): paper trading with local JSON state
  - alpaca: real paper-trading via Alpaca API

Supports two data feeds:
  - yfinance (default for local): free, no API key needed
  - alpaca: requires ALPACA_API_KEY / ALPACA_SECRET_KEY env vars

And two modes:
  - polling (default): fetches latest price every N seconds via REST
  - streaming (--stream): real-time trade-by-trade via Alpaca WebSocket

Usage:
    python app.py MSFT                              # local paper trading
    python app.py MSFT --resume                     # resume previous session
    python app.py MSFT --summary                    # print last session stats
    python app.py MSFT --figures                    # generate plots
    python app.py MSFT --strategy ou --entry 1.0    # OU strategy
    python app.py MSFT --feed alpaca --stream       # WebSocket streaming (local)
    python app.py MSFT --broker alpaca              # Alpaca paper trading
    python app.py MSFT --broker alpaca --dry-run    # observe, no orders
    python app.py MSFT --broker alpaca --profile portfolio  # different account
"""

import argparse
import json
import math
import os
import signal
import sys
import time as time_mod
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from finance_tools.backtest.engine import Portfolio, Action, ActionType
from finance_tools.broker.data_feed import YFinanceFeed, AlpacaFeed, AlpacaStreamFeed, Quote
from finance_tools.strategies.intraday import (
    IntradayChebyshevWithCooldown, IntradayOUWithCooldown,
)
from finance_tools.broker.alpaca import AlpacaBroker, OrderResult

# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Risk limits
# ---------------------------------------------------------------------------

@dataclass
class RiskLimits:
    """Risk management parameters."""
    cash_reserve_pct: float = 0.05
    max_position_pct: float = 0.90
    max_trades_per_session: int = 50


def check_risk(action, portfolio: Portfolio, risk: RiskLimits,
               trade_count: int):
    """Filter a strategy action through risk limits.

    Returns the (possibly modified) action.  May downgrade buys/sells
    or replace them with holds.
    """
    # Circuit breaker
    if trade_count >= risk.max_trades_per_session:
        return Action.hold()

    if action.action == ActionType.BUY:
        total_value = portfolio.total_value
        min_cash = total_value * risk.cash_reserve_pct
        available = max(portfolio.cash - min_cash, 0.0)

        max_equity = total_value * risk.max_position_pct
        headroom = max(max_equity - portfolio.equity, 0.0)
        available = min(available, headroom)

        if available <= 0 or portfolio.price <= 0:
            return Action.hold()

        if action.shares is not None:
            max_shares = math.floor(available / portfolio.price)
            capped = min(int(action.shares), max_shares)
            if capped < 1:
                return Action.hold()
            return Action.buy_shares(capped)
        else:
            return action

    elif action.action == ActionType.SELL:
        if portfolio.shares <= 0:
            return Action.hold()
        if action.shares is not None:
            capped = min(int(action.shares), int(portfolio.shares))
            if capped < 1:
                return Action.hold()
            return Action.sell_shares(capped)
        else:
            return action

    return action


# ---------------------------------------------------------------------------
# Alpaca bridge -- portfolio state & order execution
# ---------------------------------------------------------------------------

def fetch_portfolio_state(broker: AlpacaBroker, ticker: str,
                          price: float) -> Portfolio:
    """Build a Portfolio from Alpaca's real position + cash.

    Parameters
    ----------
    broker : AlpacaBroker
        Connected broker instance.
    ticker : str
        The ticker being traded.
    price : float
        Current market price (from data feed).

    Returns
    -------
    Portfolio with live cash and share count.
    """
    cash = broker.get_cash()
    pos = broker.get_position(ticker)
    shares = pos.qty if pos else 0
    return Portfolio(cash=cash, shares=shares, price=price)


def execute_action(broker: AlpacaBroker, ticker: str, action,
                   state: dict, dry_run: bool = False) -> OrderResult | None:
    """Submit an order to Alpaca, wait for fill, and log to state.

    In dry-run mode, logs the trade with status="dry_run" but makes no
    broker calls.

    Returns
    -------
    OrderResult if a real order was placed, None for holds or dry runs.
    """
    if action.action == ActionType.HOLD:
        return None

    shares = int(action.shares) if action.shares else 0
    if shares < 1:
        return None

    side = "buy" if action.action == ActionType.BUY else "sell"
    now_str = datetime.now().isoformat(timespec="seconds")

    if dry_run:
        state["trades"].append({
            "timestamp": now_str,
            "action": side,
            "shares": shares,
            "price": 0.0,
            "order_id": None,
            "status": "dry_run",
        })
        return None

    try:
        if side == "buy":
            result = broker.buy(ticker, shares)
        else:
            result = broker.sell(ticker, shares)

        # Poll for fill
        if result.status not in ("filled", "canceled", "expired", "rejected"):
            result = broker.wait_for_fill(result.order_id)

        state["trades"].append({
            "timestamp": now_str,
            "action": side,
            "shares": result.filled_qty if result.filled_qty else shares,
            "price": result.filled_price if result.filled_price else 0.0,
            "order_id": result.order_id,
            "status": result.status,
        })
        return result

    except Exception as e:
        state["trades"].append({
            "timestamp": now_str,
            "action": side,
            "shares": shares,
            "price": 0.0,
            "order_id": None,
            "status": "error",
            "error": str(e),
        })
        return None


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def _state_file(ticker: str, broker_mode: str = "local") -> str:
    """Path to the per-ticker state file."""
    if broker_mode == "alpaca":
        return os.path.join(SCRIPT_DIR, f".alpaca_intraday_{ticker}.json")
    return os.path.join(SCRIPT_DIR, f".intraday_{ticker}.json")


def load_state(ticker: str, broker_mode: str = "local") -> dict | None:
    """Load session state for a ticker. Returns None if not found."""
    path = _state_file(ticker, broker_mode)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_state(state: dict) -> None:
    """Save session state."""
    broker_mode = state.get("broker_mode", "local")
    path = _state_file(state["ticker"], broker_mode)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def create_initial_state(ticker: str, k: float, window: int,
                         interval: int, cooldown: int,
                         strategy_type: str = "chebyshev",
                         entry: float = 2.0,
                         max_threshold: float = 3.0,
                         broker_mode: str = "local",
                         cash: float = 10_000.0,
                         profile: str = "intraday") -> dict:
    """Create a fresh session state.

    For local broker: stores cash/shares at top level.
    For alpaca broker: Alpaca is the source of truth (no local cash/shares).
    """
    base = {
        "ticker": ticker,
        "broker_mode": broker_mode,
        "session_start": datetime.now().isoformat(timespec="seconds"),
        "params": {
            "strategy": strategy_type,
            "k_threshold": k,
            "window": window,
            "interval": interval,
            "cooldown": cooldown,
            "entry_threshold": entry,
            "max_threshold": max_threshold,
        },
        "trades": [],
        "tick_log": [],
    }

    if broker_mode == "local":
        base["cash"] = cash
        base["shares"] = 0
        base["initial_cash"] = cash
    else:
        base["profile"] = profile

    return base


# ---------------------------------------------------------------------------
# Session statistics
# ---------------------------------------------------------------------------

def compute_session_stats(state: dict) -> dict:
    """Compute end-of-session performance statistics.

    Returns dict with: pnl, pnl_pct, n_buys, n_sells, n_trades,
    sharpe, duration_minutes, last_price.
    """
    tick_log = state.get("tick_log", [])
    trades = state.get("trades", [])
    broker_mode = state.get("broker_mode", "local")

    if not tick_log:
        return {
            "pnl": 0.0, "pnl_pct": 0.0, "current_value": 0.0,
            "n_buys": 0, "n_sells": 0, "n_trades": 0,
            "sharpe": 0.0, "duration_minutes": 0.0, "last_price": 0.0,
        }

    last_price = tick_log[-1].get("price", 0.0)

    if broker_mode == "local":
        initial = state.get("initial_cash", 0.0)
        current_value = state["cash"] + state["shares"] * last_price
        pnl = current_value - initial
        pnl_pct = pnl / initial if initial > 0 else 0.0
    else:
        # Alpaca: derive from tick_log portfolio values
        initial_value = tick_log[0].get("portfolio_value", 0.0)
        final_value = tick_log[-1].get("portfolio_value", 0.0)
        current_value = final_value
        pnl = final_value - initial_value
        pnl_pct = pnl / initial_value if initial_value > 0 else 0.0

    n_buys = sum(1 for t in trades if t["action"] == "buy")
    n_sells = sum(1 for t in trades if t["action"] == "sell")

    # Approximate Sharpe from tick returns
    sharpe = 0.0
    if len(tick_log) >= 2:
        if broker_mode == "local":
            initial_val = state.get("initial_cash", 0.0)
            values = np.array(
                [t.get("portfolio_value", initial_val) for t in tick_log],
                dtype=float)
        else:
            values = np.array(
                [t.get("portfolio_value", 0) for t in tick_log], dtype=float)

        denom = np.maximum(values[:-1], 1e-8)
        returns = np.diff(values) / denom
        if len(returns) > 1 and np.std(returns) > 0:
            interval = state["params"].get("interval", 60)
            ticks_per_year = 252 * 6.5 * 3600 / interval
            sharpe = float(
                np.mean(returns) / np.std(returns) * np.sqrt(ticks_per_year))

    # Duration
    duration = 0.0
    if len(tick_log) >= 2:
        t0 = datetime.fromisoformat(tick_log[0]["timestamp"])
        t1 = datetime.fromisoformat(tick_log[-1]["timestamp"])
        duration = (t1 - t0).total_seconds() / 60.0

    return {
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "current_value": current_value,
        "n_buys": n_buys,
        "n_sells": n_sells,
        "n_trades": n_buys + n_sells,
        "sharpe": sharpe,
        "duration_minutes": duration,
        "last_price": last_price,
    }


def print_session_summary(state: dict) -> None:
    """Print a colored end-of-session report."""
    stats = compute_session_stats(state)
    broker_mode = state.get("broker_mode", "local")

    pnl_color = GREEN if stats["pnl"] >= 0 else RED
    pnl_pct_str = f"{stats['pnl_pct']:+.2%}"

    print(f"\n{BOLD}{CYAN}")
    print(f"  ╔═════════════════════════════════════╗")
    print(f"  ║   Session Summary -- {state['ticker']:<15}║")
    print(f"  ╚═════════════════════════════════════╝")
    print(f"{RESET}")

    if broker_mode == "local":
        print(f"  Initial cash:    ${state['initial_cash']:,.2f}")
        print(f"  Current value:   ${stats['current_value']:,.2f}")
        print(f"  P&L:             {pnl_color}${stats['pnl']:+,.2f} ({pnl_pct_str}){RESET}")
        print(f"  Shares held:     {state['shares']}")
        print(f"  Cash remaining:  ${state['cash']:,.2f}")
    else:
        print(f"  P&L:             {pnl_color}${stats['pnl']:+,.2f} ({pnl_pct_str}){RESET}")

    print(f"  Trades:          {stats['n_trades']} "
          f"({stats['n_buys']} buys, {stats['n_sells']} sells)")
    print(f"  Duration:        {stats['duration_minutes']:.1f} min")
    if stats["n_trades"] > 0:
        print(f"  Approx Sharpe:   {stats['sharpe']:.2f}")
    print(f"  Ticks recorded:  {len(state.get('tick_log', []))}")
    print()


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def generate_session_figures(state: dict) -> None:
    """Generate 3 publication-quality figures from session data."""
    tick_log = state.get("tick_log", [])
    trades = state.get("trades", [])
    broker_mode = state.get("broker_mode", "local")

    if len(tick_log) < 2:
        print(f"  {YELLOW}Not enough data for figures (need at least 2 ticks).{RESET}")
        return

    from finance_tools.utils.plotting import (
        setup_style, savefig, PALETTE, FIGSIZE,
    )
    import matplotlib.pyplot as plt

    setup_style()

    ticker = state["ticker"]
    fig_dir = os.path.join(SCRIPT_DIR, "figures")

    timestamps = [datetime.fromisoformat(t["timestamp"]) for t in tick_log]
    prices = np.array([t["price"] for t in tick_log])
    z_scores = np.array([t.get("z_score", 0.0) for t in tick_log])
    values = np.array([t.get("portfolio_value", 0) for t in tick_log])

    t0 = timestamps[0]
    minutes = np.array([(t - t0).total_seconds() / 60.0 for t in timestamps])

    # --- Figure 1: Price with buy/sell markers ---
    fig, ax = plt.subplots(figsize=FIGSIZE["double"])
    ax.plot(minutes, prices, color=PALETTE["blue"], lw=1.2, label="Price")

    buy_times, buy_prices, sell_times, sell_prices = [], [], [], []
    for trade in trades:
        # Skip dry-run and error trades in alpaca mode
        if trade.get("status") in ("dry_run", "error"):
            continue
        ts = datetime.fromisoformat(trade["timestamp"])
        m = (ts - t0).total_seconds() / 60.0
        if trade["action"] == "buy":
            buy_times.append(m)
            buy_prices.append(trade["price"])
        else:
            sell_times.append(m)
            sell_prices.append(trade["price"])

    if buy_times:
        ax.scatter(buy_times, buy_prices, color=PALETTE["green"],
                   marker="^", s=50, zorder=5, label="Buy")
    if sell_times:
        ax.scatter(sell_times, sell_prices, color=PALETTE["red"],
                   marker="v", s=50, zorder=5, label="Sell")

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel(f"{ticker} Price (\\$)")
    ax.set_title(f"{ticker} Intraday Price with Trades")
    ax.legend(loc="best")
    savefig(fig, os.path.join(fig_dir, "price_with_trades.png"))

    # --- Figure 2: Z-score / OU signal with threshold bands ---
    strategy_type = state["params"].get("strategy", "chebyshev")
    if strategy_type == "ou":
        threshold = state["params"].get("entry_threshold", 2.0)
        sig_label = "OU Signal ($s$)"
        thresh_label = "entry"
    else:
        threshold = state["params"]["k_threshold"]
        sig_label = "z-score"
        thresh_label = "k"

    fig, ax = plt.subplots(figsize=FIGSIZE["double"])
    ax.plot(minutes, z_scores, color=PALETTE["blue"], lw=1.0, label=sig_label)
    ax.axhline(threshold, color=PALETTE["red"], ls="--", lw=1.0,
               label=f"$+{thresh_label} = {threshold}$")
    ax.axhline(-threshold, color=PALETTE["green"], ls="--", lw=1.0,
               label=f"$-{thresh_label} = {threshold}$")
    ax.axhline(0, color=PALETTE["grey"], ls=":", lw=0.8, alpha=0.5)
    ax.fill_between(minutes, -threshold, threshold,
                    color=PALETTE["grey"], alpha=0.08)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel(sig_label)
    ax.set_title(f"{ticker} Tick {sig_label}")
    ax.legend(loc="best")
    savefig(fig, os.path.join(fig_dir, "z_score.png"))

    # --- Figure 3: Cumulative P&L ---
    if broker_mode == "local":
        initial = state.get("initial_cash", values[0])
    else:
        initial = values[0] if len(values) > 0 else 0
    pnl = values - initial

    fig, ax = plt.subplots(figsize=FIGSIZE["double"])
    color = PALETTE["green"] if pnl[-1] >= 0 else PALETTE["red"]
    ax.plot(minutes, pnl, color=color, lw=1.2)
    ax.axhline(0, color=PALETTE["grey"], ls=":", lw=0.8, alpha=0.5)
    ax.fill_between(minutes, 0, pnl, where=(pnl >= 0),
                    color=PALETTE["green"], alpha=0.15)
    ax.fill_between(minutes, 0, pnl, where=(pnl < 0),
                    color=PALETTE["red"], alpha=0.15)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Cumulative P\\&L (\\$)")
    ax.set_title(f"{ticker} Session P\\&L")
    savefig(fig, os.path.join(fig_dir, "cumulative_pnl.png"))

    print(f"  {GREEN}Generated 3 figures in {fig_dir}/{RESET}")


# ---------------------------------------------------------------------------
# Warmup -- pre-seed history with recent 1m bars
# ---------------------------------------------------------------------------

def warmup_history(feed, window: int) -> list[float]:
    """Fetch recent 1-minute bar closes to pre-seed the strategy's rolling window.

    Returns a list of close prices (at least ``window + 1`` entries if
    available) so the strategy can start making decisions on the very
    first live tick instead of waiting 5+ minutes for enough history.
    """
    try:
        lookback = max(window * 2, 60)
        hist = feed.history(lookback_minutes=lookback)
        if hist is not None and len(hist) > 0:
            prices = hist["Close"].dropna().tolist()
            return prices
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Polling loop -- LOCAL broker
# ---------------------------------------------------------------------------

def run_polling_loop_local(state: dict, feed, strategy,
                           risk: RiskLimits) -> dict:
    """Main polling loop for local broker. Fetches quotes, runs strategy,
    mutates local state for trades.

    Runs until Ctrl+C. Saves state after every tick.
    """
    interval = state["params"]["interval"]
    window = state["params"]["window"]
    ticker = state["ticker"]
    trade_count = len(state.get("trades", []))

    # Build initial tick history from logged ticks (for resume)
    tick_prices = [t["price"] for t in state.get("tick_log", [])]
    tick_times = [t["timestamp"] for t in state.get("tick_log", [])]

    # Pre-seed with recent 1m bars so the strategy can decide immediately
    if len(tick_prices) < window + 1:
        print(f"  {DIM}Warming up with recent 1-minute bars...{RESET}")
        warmup_prices = warmup_history(feed, window)
        if warmup_prices:
            tick_prices = warmup_prices + tick_prices
            n_warmup = len(warmup_prices)
            tick_times = [""] * n_warmup + tick_times
            print(f"  {DIM}Pre-seeded {n_warmup} bars -- strategy ready{RESET}")
        else:
            need = window + 1 - len(tick_prices)
            print(f"  {YELLOW}No warmup data -- strategy needs {need} more "
                  f"ticks before trading{RESET}")

    print(f"\n  {BOLD}{CYAN}Polling {ticker} every {interval}s -- "
          f"Ctrl+C to stop{RESET}\n")

    running = True

    def _handle_sigint(sig, frame):
        nonlocal running
        running = False
        print(f"\n\n  {YELLOW}Stopping...{RESET}")

    old_handler = signal.signal(signal.SIGINT, _handle_sigint)

    try:
        while running:
            try:
                quote = feed.latest()
            except Exception as e:
                print(f"  {RED}Fetch error: {e}{RESET}")
                time_mod.sleep(interval)
                continue

            price = quote.price
            now = datetime.now()
            now_str = now.isoformat(timespec="seconds")

            tick_prices.append(price)
            tick_times.append(now_str)

            history_df = pd.DataFrame({"Close": tick_prices})
            day = history_df.iloc[-1]

            portfolio = Portfolio(
                cash=state["cash"],
                shares=state["shares"],
                price=price,
            )

            action = strategy.decide(day, history_df, portfolio)
            action = check_risk(action, portfolio, risk, trade_count)

            z = strategy.compute_z(history_df)
            sig_label = "s" if state["params"].get("strategy") == "ou" else "z"
            z_str = (f"{sig_label}={z:+.2f}" if z is not None
                     else f"{sig_label}=---")

            # Execute locally
            action_str = ""
            if (action.action == ActionType.BUY and action.shares
                    and action.shares > 0):
                n = int(action.shares)
                cost = n * price
                state["shares"] += n
                state["cash"] -= cost
                trade_count += 1
                state["trades"].append({
                    "timestamp": now_str,
                    "action": "buy",
                    "shares": n,
                    "price": price,
                    "cost": round(cost, 2),
                })
                action_str = f"  {GREEN}-> BUY {n} shares{RESET}"
            elif (action.action == ActionType.SELL and action.shares
                  and action.shares > 0):
                n = int(action.shares)
                proceeds = n * price
                state["shares"] -= n
                state["cash"] += proceeds
                trade_count += 1
                state["trades"].append({
                    "timestamp": now_str,
                    "action": "sell",
                    "shares": n,
                    "price": price,
                    "proceeds": round(proceeds, 2),
                })
                action_str = f"  {RED}-> SELL {n} shares{RESET}"

            total_value = state["cash"] + state["shares"] * price

            state["tick_log"].append({
                "timestamp": now_str,
                "price": price,
                "z_score": z if z is not None else 0.0,
                "portfolio_value": total_value,
                "shares": state["shares"],
                "cash": state["cash"],
            })

            time_str = now.strftime("%H:%M:%S")
            market_tag = ("" if quote.market_open
                          else f" {DIM}(closed){RESET}")
            print(f"  [{time_str}] {ticker} ${price:.2f}  |  "
                  f"{z_str}{action_str}{market_tag}")
            print(f"  {DIM}         Shares: {state['shares']}  "
                  f"Cash: ${state['cash']:,.2f}  "
                  f"Value: ${total_value:,.2f}{RESET}")

            save_state(state)
            time_mod.sleep(interval)

    finally:
        signal.signal(signal.SIGINT, old_handler)

    return state


# ---------------------------------------------------------------------------
# Streaming loop -- LOCAL broker
# ---------------------------------------------------------------------------

def run_streaming_loop_local(state: dict, feed, strategy,
                             risk: RiskLimits) -> dict:
    """Main streaming loop for local broker. Receives aggregated ticks
    via WebSocket.

    Runs until Ctrl+C. Saves state after every tick.
    """
    window = state["params"]["window"]
    ticker = state["ticker"]
    trade_count = len(state.get("trades", []))

    tick_prices = [t["price"] for t in state.get("tick_log", [])]
    tick_times = [t["timestamp"] for t in state.get("tick_log", [])]

    if len(tick_prices) < window + 1:
        print(f"  {DIM}Warming up with recent 1-minute bars...{RESET}")
        warmup_prices = warmup_history(feed, window)
        if warmup_prices:
            tick_prices = warmup_prices + tick_prices
            n_warmup = len(warmup_prices)
            tick_times = [""] * n_warmup + tick_times
            print(f"  {DIM}Pre-seeded {n_warmup} bars -- strategy ready{RESET}")
        else:
            need = window + 1 - len(tick_prices)
            print(f"  {YELLOW}No warmup data -- strategy needs {need} more "
                  f"ticks before trading{RESET}")

    print(f"  {DIM}Starting WebSocket stream...{RESET}")
    feed.start()
    print(f"\n  {BOLD}{CYAN}Streaming {ticker} (real-time) -- "
          f"Ctrl+C to stop{RESET}\n")

    running = True

    def _handle_sigint(sig, frame):
        nonlocal running
        running = False
        print(f"\n\n  {YELLOW}Stopping...{RESET}")

    old_handler = signal.signal(signal.SIGINT, _handle_sigint)

    try:
        while running:
            tick = feed.get_tick(timeout=5.0)
            if tick is None:
                continue

            price = tick.price
            now = datetime.now()
            now_str = now.isoformat(timespec="seconds")

            tick_prices.append(price)
            tick_times.append(now_str)

            history_df = pd.DataFrame({"Close": tick_prices})
            day = history_df.iloc[-1]

            portfolio = Portfolio(
                cash=state["cash"],
                shares=state["shares"],
                price=price,
            )

            action = strategy.decide(day, history_df, portfolio)
            action = check_risk(action, portfolio, risk, trade_count)

            z = strategy.compute_z(history_df)
            sig_label = "s" if state["params"].get("strategy") == "ou" else "z"
            z_str = (f"{sig_label}={z:+.2f}" if z is not None
                     else f"{sig_label}=---")

            action_str = ""
            if (action.action == ActionType.BUY and action.shares
                    and action.shares > 0):
                n = int(action.shares)
                cost = n * price
                state["shares"] += n
                state["cash"] -= cost
                trade_count += 1
                state["trades"].append({
                    "timestamp": now_str,
                    "action": "buy",
                    "shares": n,
                    "price": price,
                    "cost": round(cost, 2),
                })
                action_str = f"  {GREEN}-> BUY {n} shares{RESET}"
            elif (action.action == ActionType.SELL and action.shares
                  and action.shares > 0):
                n = int(action.shares)
                proceeds = n * price
                state["shares"] -= n
                state["cash"] += proceeds
                trade_count += 1
                state["trades"].append({
                    "timestamp": now_str,
                    "action": "sell",
                    "shares": n,
                    "price": price,
                    "proceeds": round(proceeds, 2),
                })
                action_str = f"  {RED}-> SELL {n} shares{RESET}"

            total_value = state["cash"] + state["shares"] * price

            state["tick_log"].append({
                "timestamp": now_str,
                "price": price,
                "z_score": z if z is not None else 0.0,
                "portfolio_value": total_value,
                "shares": state["shares"],
                "cash": state["cash"],
            })

            time_str = now.strftime("%H:%M:%S")
            n_trades_str = (f" ({tick.n_trades} trades)"
                            if tick.n_trades > 1 else "")
            print(f"  [{time_str}] {ticker} ${price:.2f}{n_trades_str}  |  "
                  f"{z_str}{action_str}")
            print(f"  {DIM}         Shares: {state['shares']}  "
                  f"Cash: ${state['cash']:,.2f}  "
                  f"Value: ${total_value:,.2f}{RESET}")

            save_state(state)

    finally:
        signal.signal(signal.SIGINT, old_handler)
        feed.stop()

    return state


# ---------------------------------------------------------------------------
# Polling loop -- ALPACA broker
# ---------------------------------------------------------------------------

def run_polling_loop_alpaca(state: dict, feed, strategy, risk: RiskLimits,
                            broker: AlpacaBroker,
                            dry_run: bool = False) -> dict:
    """Main polling loop for Alpaca broker. Fetches quotes, runs strategy,
    executes via Alpaca.

    Runs until Ctrl+C. Saves state after every tick.
    """
    interval = state["params"]["interval"]
    window = state["params"]["window"]
    ticker = state["ticker"]
    trade_count = len([t for t in state.get("trades", [])
                       if t.get("status") not in ("dry_run", "error")])

    tick_prices = [t["price"] for t in state.get("tick_log", [])]

    if len(tick_prices) < window + 1:
        print(f"  {DIM}Warming up with recent 1-minute bars...{RESET}")
        warmup_prices = warmup_history(feed, window)
        if warmup_prices:
            tick_prices = warmup_prices + tick_prices
            print(f"  {DIM}Pre-seeded {len(warmup_prices)} bars -- "
                  f"strategy ready{RESET}")
        else:
            need = window + 1 - len(tick_prices)
            print(f"  {YELLOW}No warmup data -- strategy needs {need} more "
                  f"ticks before trading{RESET}")

    print(f"\n  {BOLD}{CYAN}Polling {ticker} every {interval}s -- "
          f"Ctrl+C to stop{RESET}\n")

    running = True

    def _handle_sigint(sig, frame):
        nonlocal running
        running = False
        print(f"\n\n  {YELLOW}Stopping...{RESET}")

    old_handler = signal.signal(signal.SIGINT, _handle_sigint)

    try:
        while running:
            try:
                quote = feed.latest()
            except Exception as e:
                print(f"  {RED}Fetch error: {e}{RESET}")
                time_mod.sleep(interval)
                continue

            price = quote.price
            now = datetime.now()
            now_str = now.isoformat(timespec="seconds")

            tick_prices.append(price)

            history_df = pd.DataFrame({"Close": tick_prices})
            day = history_df.iloc[-1]

            # Portfolio from Alpaca
            try:
                portfolio = fetch_portfolio_state(broker, ticker, price)
            except Exception as e:
                print(f"  {RED}Alpaca error: {e}{RESET}")
                time_mod.sleep(interval)
                continue

            action = strategy.decide(day, history_df, portfolio)
            action = check_risk(action, portfolio, risk, trade_count)

            z = strategy.compute_z(history_df)
            sig_label = ("s" if state["params"].get("strategy") == "ou"
                         else "z")
            z_str = (f"{sig_label}={z:+.2f}" if z is not None
                     else f"{sig_label}=---")

            # Execute via Alpaca (or dry-run)
            action_str = ""
            if (action.action == ActionType.BUY and action.shares
                    and action.shares > 0):
                n = int(action.shares)
                result = execute_action(broker, ticker, action, state,
                                        dry_run=dry_run)
                trade_count += 1
                tag = " (dry)" if dry_run else ""
                if result and result.status == "filled":
                    action_str = (f"  {GREEN}-> BUY {result.filled_qty} "
                                  f"@ ${result.filled_price:.2f}{tag}{RESET}")
                else:
                    action_str = f"  {GREEN}-> BUY {n} shares{tag}{RESET}"

            elif (action.action == ActionType.SELL and action.shares
                  and action.shares > 0):
                n = int(action.shares)
                result = execute_action(broker, ticker, action, state,
                                        dry_run=dry_run)
                trade_count += 1
                tag = " (dry)" if dry_run else ""
                if result and result.status == "filled":
                    action_str = (f"  {RED}-> SELL {result.filled_qty} "
                                  f"@ ${result.filled_price:.2f}{tag}{RESET}")
                else:
                    action_str = f"  {RED}-> SELL {n} shares{tag}{RESET}"

            # Refresh portfolio for logging
            try:
                portfolio = fetch_portfolio_state(broker, ticker, price)
            except Exception:
                pass
            total_value = portfolio.total_value

            state["tick_log"].append({
                "timestamp": now_str,
                "price": price,
                "z_score": z if z is not None else 0.0,
                "portfolio_value": total_value,
                "shares": int(portfolio.shares),
                "cash": portfolio.cash,
            })

            time_str = now.strftime("%H:%M:%S")
            market_tag = ("" if quote.market_open
                          else f" {DIM}(closed){RESET}")
            print(f"  [{time_str}] {ticker} ${price:.2f}  |  "
                  f"{z_str}{action_str}{market_tag}")
            print(f"  {DIM}         Shares: {int(portfolio.shares)}  "
                  f"Cash: ${portfolio.cash:,.2f}  "
                  f"Value: ${total_value:,.2f}{RESET}")

            save_state(state)
            time_mod.sleep(interval)

    finally:
        signal.signal(signal.SIGINT, old_handler)

    return state


# ---------------------------------------------------------------------------
# Streaming loop -- ALPACA broker
# ---------------------------------------------------------------------------

def run_streaming_loop_alpaca(state: dict, feed, strategy, risk: RiskLimits,
                              broker: AlpacaBroker,
                              dry_run: bool = False) -> dict:
    """Main streaming loop for Alpaca broker. Receives aggregated ticks
    via WebSocket.

    Runs until Ctrl+C. Saves state after every tick.
    """
    window = state["params"]["window"]
    ticker = state["ticker"]
    trade_count = len([t for t in state.get("trades", [])
                       if t.get("status") not in ("dry_run", "error")])

    tick_prices = [t["price"] for t in state.get("tick_log", [])]

    if len(tick_prices) < window + 1:
        print(f"  {DIM}Warming up with recent 1-minute bars...{RESET}")
        warmup_prices = warmup_history(feed, window)
        if warmup_prices:
            tick_prices = warmup_prices + tick_prices
            print(f"  {DIM}Pre-seeded {len(warmup_prices)} bars -- "
                  f"strategy ready{RESET}")
        else:
            need = window + 1 - len(tick_prices)
            print(f"  {YELLOW}No warmup data -- strategy needs {need} more "
                  f"ticks before trading{RESET}")

    print(f"  {DIM}Starting WebSocket stream...{RESET}")
    feed.start()
    print(f"\n  {BOLD}{CYAN}Streaming {ticker} (real-time) -- "
          f"Ctrl+C to stop{RESET}\n")

    running = True

    def _handle_sigint(sig, frame):
        nonlocal running
        running = False
        print(f"\n\n  {YELLOW}Stopping...{RESET}")

    old_handler = signal.signal(signal.SIGINT, _handle_sigint)

    try:
        while running:
            tick = feed.get_tick(timeout=5.0)
            if tick is None:
                continue

            price = tick.price
            now = datetime.now()
            now_str = now.isoformat(timespec="seconds")

            tick_prices.append(price)

            history_df = pd.DataFrame({"Close": tick_prices})
            day = history_df.iloc[-1]

            # Portfolio from Alpaca
            try:
                portfolio = fetch_portfolio_state(broker, ticker, price)
            except Exception as e:
                print(f"  {RED}Alpaca error: {e}{RESET}")
                continue

            action = strategy.decide(day, history_df, portfolio)
            action = check_risk(action, portfolio, risk, trade_count)

            z = strategy.compute_z(history_df)
            sig_label = ("s" if state["params"].get("strategy") == "ou"
                         else "z")
            z_str = (f"{sig_label}={z:+.2f}" if z is not None
                     else f"{sig_label}=---")

            action_str = ""
            if (action.action == ActionType.BUY and action.shares
                    and action.shares > 0):
                n = int(action.shares)
                result = execute_action(broker, ticker, action, state,
                                        dry_run=dry_run)
                trade_count += 1
                tag = " (dry)" if dry_run else ""
                if result and result.status == "filled":
                    action_str = (f"  {GREEN}-> BUY {result.filled_qty} "
                                  f"@ ${result.filled_price:.2f}{tag}{RESET}")
                else:
                    action_str = f"  {GREEN}-> BUY {n} shares{tag}{RESET}"

            elif (action.action == ActionType.SELL and action.shares
                  and action.shares > 0):
                n = int(action.shares)
                result = execute_action(broker, ticker, action, state,
                                        dry_run=dry_run)
                trade_count += 1
                tag = " (dry)" if dry_run else ""
                if result and result.status == "filled":
                    action_str = (f"  {RED}-> SELL {result.filled_qty} "
                                  f"@ ${result.filled_price:.2f}{tag}{RESET}")
                else:
                    action_str = f"  {RED}-> SELL {n} shares{tag}{RESET}"

            # Refresh portfolio
            try:
                portfolio = fetch_portfolio_state(broker, ticker, price)
            except Exception:
                pass
            total_value = portfolio.total_value

            state["tick_log"].append({
                "timestamp": now_str,
                "price": price,
                "z_score": z if z is not None else 0.0,
                "portfolio_value": total_value,
                "shares": int(portfolio.shares),
                "cash": portfolio.cash,
            })

            time_str = now.strftime("%H:%M:%S")
            n_trades_str = (f" ({tick.n_trades} trades)"
                            if tick.n_trades > 1 else "")
            print(f"  [{time_str}] {ticker} ${price:.2f}{n_trades_str}  |  "
                  f"{z_str}{action_str}")
            print(f"  {DIM}         Shares: {int(portfolio.shares)}  "
                  f"Cash: ${portfolio.cash:,.2f}  "
                  f"Value: ${total_value:,.2f}{RESET}")

            save_state(state)

    finally:
        signal.signal(signal.SIGINT, old_handler)
        feed.stop()

    return state


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Intraday Trader")
    parser.add_argument("ticker", nargs="?", default="F",
                        help="Stock ticker (default: F)")
    parser.add_argument("--broker", choices=["local", "alpaca"],
                        default="local",
                        help="Broker mode (default: local)")
    parser.add_argument("--profile", default="intraday",
                        help="Alpaca config profile (default: intraday)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Observe without submitting orders (alpaca only)")
    parser.add_argument("--cash", type=float, default=10_000.0,
                        help="Starting cash for local mode (default: $10,000)")
    parser.add_argument("--feed", choices=["yfinance", "alpaca"],
                        default=None,
                        help="Data feed (default: yfinance for local, "
                             "alpaca for alpaca broker)")
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
    parser.add_argument("--interval", type=int, default=60,
                        help="Polling interval in seconds (default: 60)")
    parser.add_argument("--cooldown", type=int, default=6,
                        help="Cooldown ticks between trades (default: 6)")
    parser.add_argument("--stream", action="store_true",
                        help="Use WebSocket streaming (requires alpaca feed)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume previous session")
    parser.add_argument("--summary", action="store_true",
                        help="Print last session stats and exit")
    parser.add_argument("--figures", action="store_true",
                        help="Generate plots from last session and exit")
    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()
    ticker = args.ticker.upper()
    broker_mode = args.broker

    # Resolve feed default: alpaca broker always uses alpaca feed
    if args.feed is not None:
        feed_choice = args.feed
    elif broker_mode == "alpaca":
        feed_choice = "alpaca"
    else:
        feed_choice = "yfinance"

    # Summary mode
    if args.summary:
        state = load_state(ticker, broker_mode)
        if state is None:
            print(f"  {RED}No session found for {ticker}.{RESET}")
            return
        print_session_summary(state)
        return

    # Figures mode
    if args.figures:
        state = load_state(ticker, broker_mode)
        if state is None:
            print(f"  {RED}No session found for {ticker}.{RESET}")
            return
        generate_session_figures(state)
        return

    # Banner
    strategy_label = ("Ornstein-Uhlenbeck" if args.strategy == "ou"
                      else "Chebyshev")
    broker_label = "Alpaca" if broker_mode == "alpaca" else "Local"
    dry_label = "  [DRY RUN]" if args.dry_run and broker_mode == "alpaca" else ""
    print(f"\n{BOLD}{CYAN}")
    print(f"  ╔═════════════════════════════════════╗")
    print(f"  ║   Intraday Trader ({broker_label:<6})          ║")
    print(f"  ║   {strategy_label + ' Mean Reversion':<34}║")
    print(f"  ╚═════════════════════════════════════╝")
    print(f"{RESET}")

    # Alpaca broker setup
    alpaca_broker = None
    if broker_mode == "alpaca":
        print(f"  {DIM}Config: ~/.alpaca/config.yaml  |  "
              f"Profile: {args.profile}{RESET}")
        try:
            alpaca_broker = AlpacaBroker(profile=args.profile)
            print(f"  {GREEN}Connected to Alpaca paper trading.{RESET}")
        except (ValueError, KeyError, FileNotFoundError) as e:
            print(f"  {RED}{e}{RESET}")
            print(f"  {DIM}Configure ~/.alpaca/config.yaml or set "
                  f"ALPACA_API_KEY / ALPACA_SECRET_KEY env vars.{RESET}")
            return

        # Market status
        try:
            if alpaca_broker.is_market_open():
                print(f"  {GREEN}Market is OPEN.{RESET}")
            else:
                print(f"  {YELLOW}Market is CLOSED. Orders queue for "
                      f"next open.{RESET}")
        except Exception:
            print(f"  {DIM}Could not check market status.{RESET}")

    # Load or create state
    if args.resume:
        state = load_state(ticker, broker_mode)
        if state is None:
            print(f"  {YELLOW}No previous session for {ticker}. "
                  f"Starting fresh.{RESET}")
            state = create_initial_state(
                ticker, args.k, args.window, args.interval, args.cooldown,
                strategy_type=args.strategy, entry=args.entry,
                max_threshold=args.max_threshold, broker_mode=broker_mode,
                cash=args.cash, profile=args.profile)
        else:
            print(f"  {GREEN}Resumed session for {ticker} "
                  f"({len(state.get('tick_log', []))} ticks, "
                  f"{len(state.get('trades', []))} trades){RESET}")
    else:
        state = create_initial_state(
            ticker, args.k, args.window, args.interval, args.cooldown,
            strategy_type=args.strategy, entry=args.entry,
            max_threshold=args.max_threshold, broker_mode=broker_mode,
            cash=args.cash, profile=args.profile)

    # Initialize strategy
    params = state["params"]
    strat_type = params.get("strategy", "chebyshev")

    if strat_type == "ou":
        strategy = IntradayOUWithCooldown(
            window=params["window"],
            entry_threshold=params.get("entry_threshold", 2.0),
            max_threshold=params.get("max_threshold", 3.0),
            cooldown_ticks=params["cooldown"],
        )
    else:
        strategy = IntradayChebyshevWithCooldown(
            window=params["window"],
            k_threshold=params["k_threshold"],
            cooldown_ticks=params["cooldown"],
        )

    # Validate streaming
    streaming = args.stream
    if streaming and feed_choice != "alpaca":
        print(f"  {RED}--stream requires alpaca feed "
              f"(use --feed alpaca or --broker alpaca){RESET}")
        return

    # Create data feed
    if streaming:
        feed = AlpacaStreamFeed(ticker)
    elif feed_choice == "alpaca":
        feed = AlpacaFeed(ticker)
    else:
        feed = YFinanceFeed(ticker)

    risk = RiskLimits()

    # Print config
    feed_label = "Alpaca" if feed_choice == "alpaca" else "yfinance"
    mode_label = "Streaming" if streaming else f"Polling ({params['interval']}s)"
    print(f"  Ticker:    {ticker}")
    print(f"  Broker:    {broker_label}")
    print(f"  Feed:      {feed_label}")
    print(f"  Mode:      {mode_label}{dry_label}")
    print(f"  Strategy:  {strategy.name}")
    if broker_mode == "local":
        print(f"  Cash:      ${state['cash']:,.2f}")
    if strat_type == "ou":
        print(f"  Entry:     {params.get('entry_threshold', 2.0)}")
        print(f"  Max thr:   {params.get('max_threshold', 3.0)}")
    else:
        print(f"  k:         {params['k_threshold']}")
    print(f"  Window:    {params['window']} ticks")
    if not streaming:
        print(f"  Interval:  {params['interval']}s")
    print(f"  Cooldown:  {params['cooldown']} ticks")

    # Run
    if broker_mode == "alpaca":
        if streaming:
            state = run_streaming_loop_alpaca(
                state, feed, strategy, risk, alpaca_broker,
                dry_run=args.dry_run)
        else:
            state = run_polling_loop_alpaca(
                state, feed, strategy, risk, alpaca_broker,
                dry_run=args.dry_run)
    else:
        if streaming:
            state = run_streaming_loop_local(state, feed, strategy, risk)
        else:
            state = run_polling_loop_local(state, feed, strategy, risk)

    # End-of-session summary
    print_session_summary(state)
    save_state(state)


if __name__ == "__main__":
    main()
