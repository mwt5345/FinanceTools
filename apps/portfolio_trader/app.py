"""
Portfolio Trader -- Interactive CLI for portfolio management.

Supports two broker backends:
  - local:  JSON state file (paper trading, full analytics)
  - alpaca: Alpaca paper trading API (real market orders)

Strategy: Inverse-Volatility Equal Weight (whole shares, Schwab-compatible).

Usage:
    python app.py                          # local broker (default)
    python app.py --broker alpaca          # Alpaca paper trading
    python app.py --broker alpaca --profile intraday
"""

import abc
import argparse
import json
import math
import os
import sys
from datetime import datetime, date

import numpy as np
import pandas as pd
import yfinance as yf

from finance_tools.backtest.portfolio import PortfolioState
from finance_tools.strategies.equal_weight import (
    compute_target_trades,
    compute_target_shares,
    compute_rebalance_trades,
    CASH_RESERVE_PCT,
)
from finance_tools.data.market import fetch_risk_free_rate, fetch_risk_free_history
from finance_tools.data.universe import TRADING_ASSISTANT_10, ALL_TICKERS
from finance_tools.broker.alpaca import AlpacaBroker, PositionInfo, OrderResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_STATE_FILE = os.path.join(SCRIPT_DIR, ".portfolio.json")
ALPACA_STATE_FILE = os.path.join(SCRIPT_DIR, ".alpaca_portfolio.json")
DEFAULT_CASH = 5000.0
ALPACA_DASHBOARD_URL = "https://app.alpaca.markets/paper/dashboard/overview"

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


# ===========================================================================
# Broker ABC + implementations
# ===========================================================================

class PortfolioBroker(abc.ABC):
    """Abstract base class for portfolio brokers."""

    @abc.abstractmethod
    def get_positions(self, tickers: list[str]) -> dict[str, int]:
        """Return {ticker: shares} for tracked tickers (0 if not held)."""
        ...

    @abc.abstractmethod
    def get_cash(self) -> float:
        """Return available cash balance."""
        ...

    @abc.abstractmethod
    def get_prices(self, tickers: list[str]) -> dict[str, float]:
        """Return {ticker: latest_price} for tracked tickers."""
        ...

    @abc.abstractmethod
    def execute_buy(self, ticker: str, shares: int, price: float) -> dict:
        """Execute a buy order. Returns result dict."""
        ...

    @abc.abstractmethod
    def execute_sell(self, ticker: str, shares: int, price: float) -> dict:
        """Execute a sell order. Returns result dict."""
        ...


class LocalBroker(PortfolioBroker):
    """JSON state file broker for paper trading.

    State is stored in a local JSON file and managed entirely in-memory
    during the session. Prices come from yfinance.
    """

    def __init__(self, state: dict, state_file: str):
        self.state = state
        self.state_file = state_file

    def get_positions(self, tickers: list[str]) -> dict[str, int]:
        positions = self.state.get("positions", {})
        return {t: int(positions.get(t, 0)) for t in tickers}

    def get_cash(self) -> float:
        return self.state["cash"]

    def get_prices(self, tickers: list[str]) -> dict[str, float]:
        """Fetch latest prices from yfinance (3-month history)."""
        prices = {}
        for t in tickers:
            try:
                ticker = yf.Ticker(t)
                hist = ticker.history(period="3mo")
                if len(hist) > 0:
                    prices[t] = float(hist["Close"].iloc[-1])
            except Exception:
                pass
        return prices

    def execute_buy(self, ticker: str, shares: int, price: float) -> dict:
        amount = round(shares * price, 2)
        self.state["positions"][ticker] = self.state["positions"].get(ticker, 0) + shares
        self.state["cash"] -= amount
        return {"action": "BUY", "ticker": ticker, "shares": shares,
                "price": price, "amount": amount, "status": "filled"}

    def execute_sell(self, ticker: str, shares: int, price: float) -> dict:
        amount = round(shares * price, 2)
        self.state["positions"][ticker] = self.state["positions"].get(ticker, 0) - shares
        self.state["cash"] += amount
        return {"action": "SELL", "ticker": ticker, "shares": shares,
                "price": price, "amount": amount, "status": "filled"}

    def save(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)


class AlpacaPortfolioBroker(PortfolioBroker):
    """Alpaca API broker for live paper trading.

    Alpaca is the source of truth for positions and cash. Prices come
    from Alpaca positions or yfinance for unheld tickers.
    """

    def __init__(self, alpaca_broker: AlpacaBroker, state: dict, state_file: str):
        self._broker = alpaca_broker
        self.state = state
        self.state_file = state_file
        # Cached Alpaca position info (set after fetch)
        self._pos_info: dict[str, PositionInfo] = {}

    def get_positions(self, tickers: list[str]) -> dict[str, int]:
        alpaca_positions = self._broker.get_positions()
        self._pos_info = alpaca_positions
        positions = {}
        for t in tickers:
            if t in alpaca_positions:
                positions[t] = alpaca_positions[t].qty
            else:
                positions[t] = 0
        return positions

    def get_cash(self) -> float:
        return self._broker.get_cash()

    def get_prices(self, tickers: list[str]) -> dict[str, float]:
        """Get prices from Alpaca positions or yfinance fallback."""
        alpaca_positions = self._broker.get_positions()
        self._pos_info = alpaca_positions
        prices = {}
        for t in tickers:
            if t in alpaca_positions:
                prices[t] = alpaca_positions[t].current_price
            else:
                try:
                    ticker = yf.Ticker(t)
                    hist = ticker.history(period="5d")
                    if len(hist) > 0:
                        prices[t] = float(hist["Close"].iloc[-1])
                    else:
                        prices[t] = 0.0
                except Exception:
                    prices[t] = 0.0
        return prices

    def execute_buy(self, ticker: str, shares: int, price: float) -> dict:
        try:
            result = self._broker.buy(ticker, shares)
            if result.status not in ("filled", "canceled", "expired", "rejected"):
                result = self._broker.wait_for_fill(result.order_id)
            return {
                "action": "BUY", "ticker": ticker, "shares": shares,
                "price": price, "amount": round(shares * price, 2),
                "status": result.status,
                "order_id": result.order_id,
                "filled_qty": result.filled_qty,
                "filled_price": result.filled_price,
            }
        except Exception as e:
            return {
                "action": "BUY", "ticker": ticker, "shares": shares,
                "price": price, "amount": round(shares * price, 2),
                "status": "error", "error": str(e),
            }

    def execute_sell(self, ticker: str, shares: int, price: float) -> dict:
        try:
            result = self._broker.sell(ticker, shares)
            if result.status not in ("filled", "canceled", "expired", "rejected"):
                result = self._broker.wait_for_fill(result.order_id)
            return {
                "action": "SELL", "ticker": ticker, "shares": shares,
                "price": price, "amount": round(shares * price, 2),
                "status": result.status,
                "order_id": result.order_id,
                "filled_qty": result.filled_qty,
                "filled_price": result.filled_price,
            }
        except Exception as e:
            return {
                "action": "SELL", "ticker": ticker, "shares": shares,
                "price": price, "amount": round(shares * price, 2),
                "status": "error", "error": str(e),
            }

    @property
    def pos_info(self) -> dict[str, PositionInfo]:
        """Cached Alpaca position info from last get_positions/get_prices call."""
        return self._pos_info

    def save(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)


# ===========================================================================
# State persistence
# ===========================================================================

def load_state(state_file: str) -> dict:
    """Load portfolio state from JSON file."""
    with open(state_file) as f:
        return json.load(f)


def save_state(state: dict, state_file: str) -> None:
    """Save portfolio state to JSON file."""
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def create_local_initial_state(cash: float, tickers: list[str]) -> dict:
    """Create a fresh local portfolio state."""
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "cash": cash,
        "positions": {t: 0.0 for t in tickers},
        "tickers": tickers,
        "total_contributed": cash,
        "last_dividend_check": date.today().isoformat(),
        "transactions": [
            {
                "date": now,
                "action": "DEPOSIT",
                "ticker": "-",
                "shares": 0,
                "price": 0,
                "amount": cash,
            }
        ],
    }


def create_alpaca_initial_state(tickers: list[str]) -> dict:
    """Create a fresh Alpaca session state (positions/cash come from Alpaca)."""
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "tickers": tickers,
        "transactions": [],
        "session_start": now,
        "last_refresh": now,
    }


# ===========================================================================
# Price fetching
# ===========================================================================

def fetch_prices(tickers: list[str]) -> tuple[dict[str, float], dict[str, pd.DataFrame]]:
    """Fetch latest prices and 3-month history for all tickers.

    Returns (prices, history) where prices is {ticker: latest_close}
    and history is {ticker: OHLCV DataFrame}.
    """
    prices = {}
    history = {}
    print(f"\n{CYAN}Fetching market data...{RESET}")
    for t in tickers:
        try:
            ticker = yf.Ticker(t)
            hist = ticker.history(period="3mo")
            if len(hist) == 0:
                print(f"  {YELLOW}Warning: no data for {t}{RESET}")
                continue
            prices[t] = float(hist["Close"].iloc[-1])
            history[t] = hist
            print(f"  {t}: ${prices[t]:.2f}")
        except Exception as e:
            print(f"  {RED}Error fetching {t}: {e}{RESET}")
    print()
    return prices, history


def fetch_history(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Fetch 3-month daily OHLCV for volatility calculation (no printing)."""
    history: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            ticker = yf.Ticker(t)
            hist = ticker.history(period="3mo")
            if len(hist) > 0:
                history[t] = hist
        except Exception:
            pass
    return history


# ===========================================================================
# Dividend detection (local only)
# ===========================================================================

def check_dividends(state: dict, history: dict[str, pd.DataFrame]) -> dict:
    """Detect and credit dividends since last check.

    Scans yfinance history for dividend payments after last_dividend_check.
    Credits cash = shares_held * dividend_per_share for each ex-date.
    """
    last_check_str = state.get("last_dividend_check")
    if last_check_str:
        last_check = date.fromisoformat(last_check_str)
    else:
        last_check = date.today()

    now = datetime.now().isoformat(timespec="seconds")
    total_div_cash = 0.0

    for t in state["tickers"]:
        shares = state["positions"].get(t, 0.0)
        if shares <= 0 or t not in history:
            continue

        hist = history[t]
        if "Dividends" not in hist.columns:
            continue

        # Filter dividend rows after last check
        div_series = hist["Dividends"]
        for idx, div_per_share in div_series.items():
            if div_per_share <= 0:
                continue
            # Normalize index to date for comparison
            if hasattr(idx, "date"):
                ex_date = idx.date()
            else:
                ex_date = pd.Timestamp(idx).date()
            if ex_date <= last_check:
                continue

            # Credit dividend
            div_cash = shares * div_per_share
            total_div_cash += div_cash
            state["cash"] += div_cash
            state["transactions"].append({
                "date": now,
                "action": "DIVIDEND",
                "ticker": t,
                "shares": 0,
                "price": round(div_per_share, 4),
                "amount": round(div_cash, 2),
            })
            print(f"  {GREEN}DIVIDEND: {t} paid ${div_per_share:.4f}/share "
                  f"x {shares:.0f} = ${div_cash:,.2f}{RESET}")

    state["last_dividend_check"] = date.today().isoformat()

    if total_div_cash > 0:
        save_state(state, LOCAL_STATE_FILE)
        print(f"  {GREEN}Total dividends credited: ${total_div_cash:,.2f}{RESET}\n")
    return state


# ===========================================================================
# Gain/loss tracking -- average cost method (local only)
# ===========================================================================

def compute_gains(transactions: list[dict],
                  prices: dict[str, float] | None = None) -> dict[str, dict]:
    """Compute per-ticker realized and unrealized gains using average cost.

    Walks the transaction list and maintains a running average cost basis.
    BUY increases position and updates avg cost; SELL realizes gain at the
    difference between sale price and current avg cost; DIVIDEND and DEPOSIT
    are ignored.

    Parameters
    ----------
    transactions : list of transaction dicts (action, ticker, shares, price, amount)
    prices : {ticker: current_price} for unrealized gain calculation.
             If None, unrealized gains are left as 0.

    Returns
    -------
    dict mapping ticker -> {
        'total_shares': float,
        'total_cost': float,
        'avg_cost': float,
        'realized_gain': float,
        'unrealized_gain': float,
        'current_price': float,
    }
    """
    if prices is None:
        prices = {}

    holdings: dict[str, dict] = {}

    for tx in transactions:
        action = tx["action"]
        ticker = tx.get("ticker", "-")

        if action not in ("BUY", "SELL") or ticker == "-":
            continue

        if ticker not in holdings:
            holdings[ticker] = {
                "total_shares": 0.0,
                "total_cost": 0.0,
                "avg_cost": 0.0,
                "realized_gain": 0.0,
            }

        h = holdings[ticker]

        if action == "BUY":
            shares = tx["shares"]
            cost = shares * tx["price"]
            h["total_cost"] += cost
            h["total_shares"] += shares
            h["avg_cost"] = h["total_cost"] / h["total_shares"] if h["total_shares"] > 0 else 0.0

        elif action == "SELL":
            shares = tx["shares"]
            sell_price = tx["price"]
            avg = h["avg_cost"]
            h["realized_gain"] += (sell_price - avg) * shares
            h["total_cost"] -= avg * shares
            h["total_shares"] -= shares
            h["avg_cost"] = h["total_cost"] / h["total_shares"] if h["total_shares"] > 0 else 0.0

    # Build final result with unrealized gains
    result = {}
    all_tickers = set(holdings.keys()) | set(prices.keys())
    for ticker in sorted(all_tickers):
        h = holdings.get(ticker, {
            "total_shares": 0.0,
            "total_cost": 0.0,
            "avg_cost": 0.0,
            "realized_gain": 0.0,
        })
        current = prices.get(ticker, 0.0)
        if h["total_shares"] > 0 and current > 0:
            unrealized = (current - h["avg_cost"]) * h["total_shares"]
        else:
            unrealized = 0.0
        result[ticker] = {
            "total_shares": h["total_shares"],
            "total_cost": h["total_cost"],
            "avg_cost": h["avg_cost"],
            "realized_gain": h["realized_gain"],
            "unrealized_gain": unrealized,
            "current_price": current,
        }

    return result


# ===========================================================================
# Sell / Withdraw
# ===========================================================================

def compute_proportional_sells(positions: dict[str, float],
                               prices: dict[str, float],
                               target_amount: float) -> list[dict]:
    """Compute whole-share sells to raise ~$target_amount, proportional to holdings.

    Pure function -- no side effects.

    Algorithm:
      1. Each ticker contributes target * (ticker_equity / total_equity), floored
         to whole shares.
      2. If flooring leaves a shortfall, redistribute leftover budget to remaining
         tickers (greedy, most-underweight first).

    Returns list of dicts: {action, ticker, shares, price, amount}.
    """
    if target_amount <= 0:
        return []

    # Active positions only (shares > 0 and price available)
    active = {t: int(s) for t, s in positions.items()
              if s > 0 and prices.get(t, 0) > 0}
    if not active:
        return []

    total_equity = sum(active[t] * prices[t] for t in active)
    if total_equity <= 0:
        return []

    # Cap target at total equity (sell everything)
    target = min(target_amount, total_equity)

    # Step 1: proportional floor allocation
    sell_shares: dict[str, int] = {}
    for t in active:
        fraction = active[t] * prices[t] / total_equity
        ideal_shares = target * fraction / prices[t]
        sell_shares[t] = min(math.floor(ideal_shares), active[t])

    # Step 2: greedy redistribution of shortfall
    raised = sum(sell_shares[t] * prices[t] for t in sell_shares)
    shortfall = target - raised

    while shortfall > 0:
        candidates = [t for t in active
                      if sell_shares[t] < active[t]
                      and prices[t] <= shortfall + 0.01]
        if not candidates:
            break

        def _gap(t):
            ideal = target * (active[t] * prices[t] / total_equity)
            actual = sell_shares[t] * prices[t]
            return ideal - actual

        candidates.sort(key=lambda t: -_gap(t))
        best = candidates[0]
        sell_shares[best] += 1
        shortfall -= prices[best]

    # Build result
    trades = []
    for t in sorted(sell_shares):
        n = sell_shares[t]
        if n > 0:
            trades.append({
                "action": "SELL",
                "ticker": t,
                "shares": n,
                "price": prices[t],
                "amount": round(n * prices[t], 2),
            })
    return trades


def sell_shares_local(state: dict, prices: dict[str, float]) -> dict:
    """Interactive sell/withdraw sub-menu for local broker."""
    positions = state["positions"]
    tickers = state["tickers"]

    # Check for any positions to sell
    has_shares = any(positions.get(t, 0) > 0 for t in tickers)
    if not has_shares:
        print(f"\n  {YELLOW}No shares to sell.{RESET}\n")
        return state

    print(f"\n  {BOLD}Sell / Withdraw{RESET}")
    print(f"  {CYAN}a.{RESET} Proportional withdrawal (raise $X)")
    print(f"  {CYAN}b.{RESET} Sell specific ticker")
    print(f"  {CYAN}c.{RESET} Cancel")
    choice = input(f"  > ").strip().lower()

    if choice == "a":
        # Proportional withdrawal
        while True:
            raw = input(f"\n  Target amount to raise: $").strip()
            try:
                amount = float(raw.replace(",", "").replace("$", ""))
                if amount <= 0:
                    print(f"  {RED}Please enter a positive amount.{RESET}")
                    continue
                break
            except ValueError:
                print(f"  {RED}Invalid number. Try again.{RESET}")

        trades = compute_proportional_sells(positions, prices, amount)
        if not trades:
            print(f"  {YELLOW}No sells possible.{RESET}\n")
            return state

        actual = sum(t["amount"] for t in trades)
        print(f"\n  {BOLD}PROPOSED SELLS{RESET}")
        print(f"  Target: ${amount:,.2f}  |  Actual: ${actual:,.2f}")
        print(f"  {'─' * 50}")
        for trade in trades:
            print(f"  {RED}SELL {trade['shares']:>5.0f} shares of "
                  f"{trade['ticker']:<5} @ ${trade['price']:<9.2f} "
                  f"(${trade['amount']:>10,.2f}){RESET}")

        confirm = input(f"\n  Execute? (y/N): ").strip().lower()
        if confirm not in ("y", "yes"):
            print(f"  {DIM}Cancelled.{RESET}\n")
            return state

        now = datetime.now().isoformat(timespec="seconds")
        for trade in trades:
            t = trade["ticker"]
            state["positions"][t] = state["positions"].get(t, 0) - trade["shares"]
            state["cash"] += trade["amount"]
            state["transactions"].append({
                "date": now,
                "action": "SELL",
                "ticker": t,
                "shares": trade["shares"],
                "price": round(trade["price"], 2),
                "amount": trade["amount"],
            })
            print(f"  {RED}SOLD {trade['shares']:.0f} shares of {t} "
                  f"for ${trade['amount']:,.2f}{RESET}")

        save_state(state, LOCAL_STATE_FILE)
        print(f"\n  {GREEN}Sells executed. Cash: ${state['cash']:,.2f}{RESET}\n")
        return state

    elif choice == "b":
        # Sell specific ticker
        held = [(t, int(positions[t])) for t in tickers
                if positions.get(t, 0) > 0]
        if not held:
            print(f"  {YELLOW}No positions to sell.{RESET}\n")
            return state

        print(f"\n  {BOLD}Current positions:{RESET}")
        for i, (t, sh) in enumerate(held):
            price = prices.get(t, 0)
            val = sh * price
            print(f"  {CYAN}{i + 1}.{RESET} {t:<6} {sh:>5} shares  "
                  f"(${val:,.2f} @ ${price:.2f})")

        raw = input(f"\n  Pick ticker (1-{len(held)}): ").strip()
        try:
            idx = int(raw) - 1
            if not (0 <= idx < len(held)):
                raise ValueError
        except ValueError:
            print(f"  {RED}Invalid selection.{RESET}\n")
            return state

        ticker, max_shares = held[idx]
        price = prices.get(ticker, 0)

        raw = input(f"  Shares to sell (max {max_shares}): ").strip()
        try:
            sell_n = int(raw)
            if sell_n <= 0:
                print(f"  {RED}Must sell at least 1 share.{RESET}\n")
                return state
            sell_n = min(sell_n, max_shares)
        except ValueError:
            print(f"  {RED}Invalid number.{RESET}\n")
            return state

        amount = round(sell_n * price, 2)
        confirm = input(f"  Sell {sell_n} shares of {ticker} for ${amount:,.2f}? (y/N): ").strip().lower()
        if confirm not in ("y", "yes"):
            print(f"  {DIM}Cancelled.{RESET}\n")
            return state

        now = datetime.now().isoformat(timespec="seconds")
        state["positions"][ticker] = state["positions"].get(ticker, 0) - sell_n
        state["cash"] += amount
        state["transactions"].append({
            "date": now,
            "action": "SELL",
            "ticker": ticker,
            "shares": sell_n,
            "price": round(price, 2),
            "amount": amount,
        })
        save_state(state, LOCAL_STATE_FILE)
        print(f"  {RED}SOLD {sell_n} shares of {ticker} for ${amount:,.2f}{RESET}")
        print(f"  {GREEN}Cash: ${state['cash']:,.2f}{RESET}\n")
        return state

    else:
        print(f"  {DIM}Cancelled.{RESET}\n")
        return state


def sell_shares_alpaca(alpaca_broker: AlpacaBroker, state: dict,
                       positions: dict[str, int],
                       prices: dict[str, float]) -> bool:
    """Interactive sell sub-menu for Alpaca broker. Submits real sell orders.

    Returns True if any trades were executed (caches should be invalidated).
    """
    held = [(t, positions[t]) for t in sorted(positions)
            if positions.get(t, 0) > 0]
    if not held:
        print(f"\n  {YELLOW}No shares to sell.{RESET}\n")
        return False

    print(f"\n  {BOLD}Sell Shares{RESET}")
    print(f"  {CYAN}a.{RESET} Proportional sell (raise $X)")
    print(f"  {CYAN}b.{RESET} Sell specific ticker")
    print(f"  {CYAN}c.{RESET} Cancel")
    choice = input(f"  > ").strip().lower()

    if choice == "a":
        while True:
            raw = input(f"\n  Target amount to raise: $").strip()
            try:
                amount = float(raw.replace(",", "").replace("$", ""))
                if amount <= 0:
                    print(f"  {RED}Please enter a positive amount.{RESET}")
                    continue
                break
            except ValueError:
                print(f"  {RED}Invalid number. Try again.{RESET}")

        trades = compute_proportional_sells(positions, prices, amount)
        if not trades:
            print(f"  {YELLOW}No sells possible.{RESET}\n")
            return False

        actual = sum(t["amount"] for t in trades)
        print(f"\n  {BOLD}PROPOSED SELLS{RESET}")
        print(f"  Target: ${amount:,.2f}  |  Actual: ${actual:,.2f}")
        print(f"  {'─' * 50}")
        for trade in trades:
            print(f"  {RED}SELL {trade['shares']:>5} shares of "
                  f"{trade['ticker']:<5} @ ${trade['price']:<9.2f} "
                  f"(${trade['amount']:>10,.2f}){RESET}")

        confirm = input(f"\n  Execute? (y/N): ").strip().lower()
        if confirm not in ("y", "yes"):
            print(f"  {DIM}Cancelled.{RESET}\n")
            return False

        print()
        results = execute_alpaca_trades(alpaca_broker, trades, state)
        save_state(state, ALPACA_STATE_FILE)
        filled = sum(1 for r in results if r.status == "filled")
        print(f"\n  {GREEN}{filled}/{len(results)} sell orders filled.{RESET}\n")
        return True

    elif choice == "b":
        print(f"\n  {BOLD}Current positions:{RESET}")
        for i, (t, qty) in enumerate(held):
            price = prices.get(t, 0)
            val = qty * price
            print(f"  {CYAN}{i + 1}.{RESET} {t:<6} {qty:>5} shares  "
                  f"(${val:,.2f} @ ${price:.2f})")

        raw = input(f"\n  Pick ticker (1-{len(held)}): ").strip()
        try:
            idx = int(raw) - 1
            if not (0 <= idx < len(held)):
                raise ValueError
        except ValueError:
            print(f"  {RED}Invalid selection.{RESET}\n")
            return False

        ticker, max_shares = held[idx]
        price = prices.get(ticker, 0)

        raw = input(f"  Shares to sell (max {max_shares}): ").strip()
        try:
            sell_n = int(raw)
            if sell_n <= 0:
                print(f"  {RED}Must sell at least 1 share.{RESET}\n")
                return False
            sell_n = min(sell_n, max_shares)
        except ValueError:
            print(f"  {RED}Invalid number.{RESET}\n")
            return False

        amount = round(sell_n * price, 2)
        confirm = input(
            f"  Sell {sell_n} shares of {ticker} "
            f"(~${amount:,.2f})? (y/N): ").strip().lower()
        if confirm not in ("y", "yes"):
            print(f"  {DIM}Cancelled.{RESET}\n")
            return False

        trades = [{"action": "SELL", "ticker": ticker,
                   "shares": sell_n, "price": price, "amount": amount}]
        print()
        results = execute_alpaca_trades(alpaca_broker, trades, state)
        save_state(state, ALPACA_STATE_FILE)
        if results and results[0].status == "filled":
            print(f"\n  {RED}Sold {results[0].filled_qty} shares of {ticker} "
                  f"@ ${results[0].filled_price:.2f}{RESET}\n")
        else:
            print()
        return True

    else:
        print(f"  {DIM}Cancelled.{RESET}\n")
        return False


# ===========================================================================
# Alpaca trade execution
# ===========================================================================

def execute_alpaca_trades(broker: AlpacaBroker,
                          trades: list[dict],
                          state: dict) -> list[OrderResult]:
    """Submit trades to Alpaca and poll for fills.

    Sells are executed before buys. Results are logged to state.

    Returns list of OrderResults.
    """
    sells = [t for t in trades if t["action"] == "SELL"]
    buys = [t for t in trades if t["action"] == "BUY"]
    ordered = sells + buys
    results = []

    for trade in ordered:
        ticker = trade["ticker"]
        qty = trade["shares"]
        action = trade["action"]
        now = datetime.now().isoformat(timespec="seconds")

        print(f"  Submitting {action} {qty} {ticker}...", end=" ", flush=True)

        try:
            if action == "BUY":
                result = broker.buy(ticker, qty)
            else:
                result = broker.sell(ticker, qty)

            # Poll for fill
            if result.status not in ("filled", "canceled", "expired", "rejected"):
                result = broker.wait_for_fill(result.order_id)

            # Log to local state
            state["transactions"].append({
                "date": now,
                "action": action,
                "ticker": ticker,
                "qty": qty,
                "order_id": result.order_id,
                "status": result.status,
                "filled_qty": result.filled_qty,
                "filled_price": result.filled_price,
            })

            if result.status == "filled":
                color = GREEN if action == "BUY" else RED
                print(f"{color}FILLED {result.filled_qty} @ "
                      f"${result.filled_price:.2f}{RESET}")
            else:
                print(f"{YELLOW}{result.status}{RESET}")

            results.append(result)

        except Exception as e:
            print(f"{RED}ERROR: {e}{RESET}")
            state["transactions"].append({
                "date": now,
                "action": action,
                "ticker": ticker,
                "qty": qty,
                "order_id": None,
                "status": "error",
                "filled_qty": 0,
                "filled_price": None,
                "error": str(e),
            })

    return results


# ===========================================================================
# Portfolio performance stats -- Sharpe ratio etc. (local only)
# ===========================================================================

def fetch_full_history(tickers: list[str],
                       start_date: str) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV from *start_date* to today for Sharpe calculation.

    Unlike fetch_prices (3 months), this covers the full portfolio lifetime.
    """
    history: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            ticker = yf.Ticker(t)
            hist = ticker.history(start=start_date)
            if len(hist) > 0:
                # Normalize timezone
                if hasattr(hist.index, "tz") and hist.index.tz is not None:
                    hist.index = hist.index.tz_localize(None)
                hist.index = pd.to_datetime(hist.index).normalize()
                hist = hist[~hist.index.duplicated(keep="first")]
                history[t] = hist
        except Exception:
            pass
    return history


def compute_portfolio_stats(transactions: list[dict],
                            tickers: list[str],
                            history: dict[str, pd.DataFrame],
                            rf_rate: float = 0.0,
                            rf_history: pd.Series | None = None) -> dict | None:
    """Replay transactions against daily prices to compute performance stats.

    Parameters
    ----------
    rf_rate : annualized risk-free rate as decimal (constant fallback).
    rf_history : daily annualized rf rate Series.  When provided, daily
                 excess returns are computed using the actual historical
                 rate on each day.

    Returns dict with keys: sharpe, ann_return, ann_vol, rf_rate, n_days,
    max_drawdown.  Returns None if fewer than 2 trading days of data.
    """
    if not history or not transactions:
        return None

    # Find the union of all trading dates across tickers
    all_dates = sorted(set().union(*(set(df.index) for df in history.values())))
    if len(all_dates) < 2:
        return None

    # Replay transactions day by day
    cash = 0.0
    positions: dict[str, int] = {t: 0 for t in tickers}
    total_contributed = 0.0
    tx_idx = 0

    daily_values = []
    daily_contributions = []

    # Sort transactions by date
    sorted_txns = sorted(transactions, key=lambda tx: tx["date"])

    for day in all_dates:
        contribution = 0.0

        # Apply all transactions up to and including this day
        while tx_idx < len(sorted_txns):
            tx_date_str = sorted_txns[tx_idx]["date"]
            tx_date = pd.Timestamp(tx_date_str[:10]).normalize()
            if tx_date > day:
                break
            tx = sorted_txns[tx_idx]
            action = tx["action"]
            ticker = tx.get("ticker", "-")

            if action == "DEPOSIT":
                cash += tx["amount"]
                total_contributed += tx["amount"]
                contribution += tx["amount"]
            elif action == "DIVIDEND":
                cash += tx["amount"]
            elif action == "BUY" and ticker != "-":
                spend = tx["shares"] * tx["price"]
                cash -= spend
                positions[ticker] = positions.get(ticker, 0) + int(tx["shares"])
            elif action == "SELL" and ticker != "-":
                proceeds = tx["shares"] * tx["price"]
                cash += proceeds
                positions[ticker] = positions.get(ticker, 0) - int(tx["shares"])

            tx_idx += 1

        # Compute portfolio value using today's prices
        equity = 0.0
        for t in tickers:
            if t in history and day in history[t].index:
                price = float(history[t].loc[day, "Close"])
                equity += positions.get(t, 0) * price

        daily_values.append(cash + equity)
        daily_contributions.append(contribution)

    if len(daily_values) < 2:
        return None

    values = np.array(daily_values, dtype=float)
    contributions = np.array(daily_contributions, dtype=float)
    n_days = len(values)

    # Annualized return (total return, annualized)
    if total_contributed <= 0:
        return None
    total_ret = values[-1] / total_contributed - 1
    ann_ret = (1 + total_ret) ** (252 / n_days) - 1

    # Daily returns (strip contributions to isolate market returns)
    market_only = values[1:] - contributions[1:]
    prev_total = values[:-1]
    # Avoid division by zero
    mask = prev_total > 0
    if mask.sum() < 2:
        return None
    daily_rets = (market_only[mask] / prev_total[mask]) - 1

    # Daily excess returns for Sharpe
    if rf_history is not None and len(rf_history) > 0:
        # Build a daily rf series aligned to all_dates
        rf_aligned = rf_history.reindex(all_dates).ffill().bfill().fillna(0.0)
        rf_daily_vals = rf_aligned.values[1:]  # skip first day (no return)
        daily_rf = rf_daily_vals[mask] / 252
        daily_excess = daily_rets - daily_rf
        avg_rf = float(rf_aligned.mean())
    else:
        daily_excess = daily_rets - rf_rate / 252
        avg_rf = rf_rate

    ann_vol = float(np.std(daily_rets, ddof=1) * np.sqrt(252))
    if len(daily_excess) >= 2 and np.std(daily_excess, ddof=1) > 0:
        sharpe = float(np.mean(daily_excess) / np.std(daily_excess, ddof=1) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Max drawdown
    cummax = np.maximum.accumulate(values)
    drawdowns = (values - cummax) / np.where(cummax > 0, cummax, 1.0)
    max_dd = float(np.min(drawdowns))

    return {
        "sharpe": round(sharpe, 2),
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "rf_rate": avg_rf,
        "n_days": n_days,
        "max_drawdown": max_dd,
    }


# ===========================================================================
# Display helpers
# ===========================================================================

def print_header(broker_name: str):
    """Print the app banner."""
    print(f"\n{BOLD}{CYAN}")
    print("  +=====================================+")
    print("  |   Portfolio Trader                   |")
    print("  |   Inv-Vol Equal Weight Strategy      |")
    print(f"  |   Broker: {broker_name:<25s} |")
    print("  +=====================================+")
    print(f"{RESET}")


def print_local_menu():
    """Print the main menu for local broker."""
    print(f"  {BOLD}What would you like to do?{RESET}")
    print(f"  {CYAN}1.{RESET} View portfolio")
    print(f"  {CYAN}2.{RESET} Get trade suggestions")
    print(f"  {CYAN}3.{RESET} Deposit cash")
    print(f"  {CYAN}4.{RESET} Sell / Withdraw")
    print(f"  {CYAN}5.{RESET} Transaction history")
    print(f"  {CYAN}6.{RESET} View gains/losses")
    print(f"  {CYAN}7.{RESET} Refresh prices")
    print(f"  {CYAN}8.{RESET} Undo last action")
    print(f"  {CYAN}9.{RESET} Save & exit")
    print()


def print_alpaca_menu():
    """Print the main menu for Alpaca broker."""
    print(f"  {BOLD}What would you like to do?{RESET}")
    print(f"  {CYAN}1.{RESET} View portfolio")
    print(f"  {CYAN}2.{RESET} Get trade suggestions")
    print(f"  {CYAN}3.{RESET} Execute suggested trades")
    print(f"  {CYAN}4.{RESET} Sell shares")
    print(f"  {CYAN}5.{RESET} Deposit cash")
    print(f"  {CYAN}6.{RESET} Refresh prices")
    print(f"  {CYAN}7.{RESET} Transaction log")
    print(f"  {CYAN}8.{RESET} Market status")
    print(f"  {CYAN}9.{RESET} Save & exit")
    print()


def view_portfolio_local(state: dict, prices: dict[str, float] | None,
                         sharpe_stats: dict | None = None):
    """Display current portfolio positions and value (local broker)."""
    tickers = state["tickers"]
    positions = state["positions"]
    cash = state["cash"]

    # Compute gains for the G/L column
    gains = compute_gains(state.get("transactions", []), prices)

    # Compute total for allocation percentages
    total_equity = 0.0
    for t in tickers:
        shares = positions.get(t, 0.0)
        price = prices.get(t, 0.0) if prices else 0.0
        total_equity += shares * price
    total_val = cash + total_equity

    print(f"\n  {BOLD}{'Ticker':<8} {'Shares':>8} {'Price':>10} {'Value':>12} {'G/L':>10} {'Alloc':>8}{RESET}")
    print(f"  {'─' * 60}")

    total_gl = 0.0
    for t in tickers:
        shares = positions.get(t, 0.0)
        price = prices.get(t, 0.0) if prices else 0.0
        value = shares * price
        alloc = value / total_val if total_val > 0 else 0.0
        alloc_str = f"{alloc:.1%}"

        # Per-ticker unrealized G/L
        g = gains.get(t, {})
        unrealized = g.get("unrealized_gain", 0.0)
        total_gl += unrealized
        gl_color = GREEN if unrealized >= 0 else RED
        gl_str = f"{gl_color}{'$' + f'{unrealized:,.2f}':>10}{RESET}"

        if shares > 0:
            color = GREEN
        else:
            color = DIM
            gl_str = f"{DIM}{'':>10}{RESET}"
        print(f"  {color}{t:<8} {shares:>8.0f} {'$' + f'{price:.2f}':>10} "
              f"{'$' + f'{value:,.2f}':>12}{RESET} {gl_str} {color}{alloc_str:>8}{RESET}")

    print(f"  {'─' * 60}")
    cash_alloc = cash / total_val if total_val > 0 else 1.0
    print(f"  {'Cash':<8} {'':>8} {'':>10} {CYAN}{'$' + f'{cash:,.2f}':>12}{RESET} {'':>10} {cash_alloc:.1%}")
    gl_total_color = GREEN if total_gl >= 0 else RED
    print(f"  {BOLD}{'Total':<8} {'':>8} {'':>10} {'$' + f'{total_val:,.2f}':>12}{RESET} "
          f"{gl_total_color}{'$' + f'{total_gl:,.2f}':>10}{RESET}")
    contributed = state.get("total_contributed", cash)
    gain = total_val - contributed
    gain_pct = gain / contributed if contributed > 0 else 0.0
    gain_color = GREEN if gain >= 0 else RED
    print(f"  Contributed: ${contributed:,.2f}  |  P&L: {gain_color}${gain:,.2f} ({gain_pct:+.1%}){RESET}")

    # Sharpe ratio line
    if sharpe_stats is not None:
        ann_ret_pct = f"{sharpe_stats['ann_return']:.1%}"
        ann_vol_pct = f"{sharpe_stats['ann_vol']:.1%}"
        rf_pct = f"{sharpe_stats.get('rf_rate', 0):.1%}"
        line = (f"  Sharpe: {BOLD}{sharpe_stats['sharpe']:.2f}{RESET}  |  "
                f"Ann. Return: {ann_ret_pct}  |  Ann. Vol: {ann_vol_pct}  |  "
                f"Rf: {rf_pct}")
        if sharpe_stats["n_days"] < 30:
            line += f"  {YELLOW}(preliminary, {sharpe_stats['n_days']} days){RESET}"
        print(line)
    elif any(positions.get(t, 0) > 0 for t in tickers):
        print(f"  {DIM}Sharpe: needs 2+ trading days of data{RESET}")

    print()


def display_portfolio_alpaca(positions: dict[str, int],
                             prices: dict[str, float],
                             cash: float,
                             tickers: list[str],
                             pos_info: dict[str, PositionInfo] | None = None):
    """Display current portfolio positions, value, and P&L (Alpaca broker)."""
    total_equity = sum(positions.get(t, 0) * prices.get(t, 0)
                       for t in tickers)
    total_val = cash + total_equity

    print(f"\n  {BOLD}{'Ticker':<7} {'Shares':>6} {'Avg Cost':>9} "
          f"{'Price':>9} {'Value':>11} {'P&L':>10} {'Alloc':>7}{RESET}")
    print(f"  {'─' * 63}")

    total_pl = 0.0
    for t in tickers:
        shares = positions.get(t, 0)
        price = prices.get(t, 0)
        value = shares * price
        alloc = value / total_val if total_val > 0 else 0.0
        alloc_str = f"{alloc:.1%}"

        # P&L from Alpaca position info
        info = pos_info.get(t) if pos_info else None
        if info and shares > 0:
            avg_cost = info.avg_entry_price
            pl = info.unrealized_pl
            total_pl += pl
            avg_str = f"${avg_cost:.2f}"
            pl_color = GREEN if pl >= 0 else RED
            pl_str = f"{pl_color}${pl:>+,.2f}{RESET}"
        else:
            avg_str = "-"
            pl_str = ""

        if shares > 0:
            color = GREEN
        else:
            color = DIM
            pl_str = ""
        print(f"  {color}{t:<7} {shares:>6} {avg_str:>9} "
              f"{'$' + f'{price:.2f}':>9} "
              f"{'$' + f'{value:,.2f}':>11}{RESET} "
              f"{pl_str:>10} {color}{alloc_str:>7}{RESET}")

    print(f"  {'─' * 63}")
    cash_alloc = cash / total_val if total_val > 0 else 1.0
    print(f"  {'Cash':<7} {'':>6} {'':>9} {'':>9} "
          f"{CYAN}{'$' + f'{cash:,.2f}':>11}{RESET} {'':>10} {cash_alloc:.1%}")
    pl_color = GREEN if total_pl >= 0 else RED
    pl_total_str = f"{pl_color}${total_pl:>+,.2f}{RESET}" if pos_info else ""
    print(f"  {BOLD}{'Total':<7} {'':>6} {'':>9} {'':>9} "
          f"{'$' + f'{total_val:,.2f}':>11}{RESET} {pl_total_str}")
    print()


def show_transaction_history(state: dict):
    """Display transaction log (local broker format)."""
    txns = state.get("transactions", [])
    if not txns:
        print(f"\n  {DIM}No transactions yet.{RESET}\n")
        return

    print(f"\n  {BOLD}{'Date':<22} {'Action':<10} {'Ticker':<8} {'Shares':>8} {'Price':>10} {'Amount':>12}{RESET}")
    print(f"  {'─' * 72}")

    for tx in txns:
        action = tx["action"]
        if action == "BUY":
            color = GREEN
        elif action == "SELL":
            color = RED
        elif action == "DEPOSIT":
            color = CYAN
        elif action == "DIVIDEND":
            color = YELLOW
        else:
            color = RESET

        shares_str = f"{tx['shares']:.0f}" if tx["shares"] > 0 else "-"
        price_str = f"${tx['price']:.2f}" if tx["price"] > 0 else "-"
        amount_str = f"${tx['amount']:,.2f}"

        print(f"  {color}{tx['date']:<22} {action:<10} {tx['ticker']:<8} "
              f"{shares_str:>8} {price_str:>10} {amount_str:>12}{RESET}")

    print()


def show_alpaca_transaction_log(state: dict):
    """Display transaction log (Alpaca broker format)."""
    txns = state.get("transactions", [])
    if not txns:
        print(f"\n  {DIM}No transactions logged this session.{RESET}\n")
        return

    print(f"\n  {BOLD}{'Date':<22} {'Action':<6} {'Ticker':<8} "
          f"{'Shares':>8} {'Price':>10} {'Status':<10}{RESET}")
    print(f"  {'─' * 68}")

    for tx in txns:
        action = tx["action"]
        color = GREEN if action == "BUY" else RED
        price_str = f"${tx['filled_price']:.2f}" if tx.get("filled_price") else "-"
        print(f"  {color}{tx['date']:<22} {action:<6} {tx['ticker']:<8} "
              f"{tx['qty']:>8} {price_str:>10} {tx['status']:<10}{RESET}")

    print()


def view_gains(state: dict, prices: dict[str, float] | None):
    """Display per-ticker realized and unrealized gains/losses (local only)."""
    gains = compute_gains(state.get("transactions", []), prices)

    if not gains:
        print(f"\n  {DIM}No positions to show.{RESET}\n")
        return

    print(f"\n  {BOLD}{'Ticker':<8} {'Shares':>8} {'Avg Cost':>10} {'Current':>10} "
          f"{'Unrealized':>12} {'Realized':>12}{RESET}")
    print(f"  {'─' * 64}")

    total_unrealized = 0.0
    total_realized = 0.0

    for ticker, g in gains.items():
        shares = g["total_shares"]
        avg_cost = g["avg_cost"]
        current = g["current_price"]
        unrealized = g["unrealized_gain"]
        realized = g["realized_gain"]
        total_unrealized += unrealized
        total_realized += realized

        # Color for unrealized
        u_color = GREEN if unrealized >= 0 else RED
        r_color = GREEN if realized >= 0 else RED

        if shares > 0:
            row_color = ""
        else:
            row_color = DIM

        avg_str = f"${avg_cost:.2f}" if avg_cost > 0 else "-"
        cur_str = f"${current:.2f}" if current > 0 else "-"

        # Only show rows with shares or realized gains
        if shares <= 0 and abs(realized) < 0.005:
            continue

        print(f"  {row_color}{ticker:<8} {shares:>8.0f} {avg_str:>10} {cur_str:>10} "
              f"{u_color}{'$' + f'{unrealized:,.2f}':>12}{RESET}{row_color} "
              f"{r_color}{'$' + f'{realized:,.2f}':>12}{RESET}")

    print(f"  {'─' * 64}")
    total_gl = total_unrealized + total_realized
    t_color = GREEN if total_gl >= 0 else RED
    u_total_color = GREEN if total_unrealized >= 0 else RED
    r_total_color = GREEN if total_realized >= 0 else RED
    print(f"  {BOLD}{'Total':<8} {'':>8} {'':>10} {'':>10} "
          f"{u_total_color}{'$' + f'{total_unrealized:,.2f}':>12}{RESET} "
          f"{r_total_color}{'$' + f'{total_realized:,.2f}':>12}{RESET}")
    print(f"  {BOLD}{'Total G/L':>48} {t_color}{'$' + f'{total_gl:,.2f}':>12}{RESET}")
    print()


def display_suggestions(trades: list[dict]):
    """Display proposed trades with dollar amounts."""
    if not trades:
        print(f"  {GREEN}Portfolio is balanced -- no trades needed.{RESET}\n")
        return

    print(f"\n  {BOLD}SUGGESTED TRADES{RESET}")
    print(f"  {'─' * 55}")
    for i, trade in enumerate(trades):
        action = trade["action"]
        color = GREEN if action == "BUY" else RED
        print(f"  {color}{i + 1}. {action:<5} {trade['shares']:>5} shares of "
              f"{trade['ticker']:<5} @ ${trade['price']:<9.2f} "
              f"(${trade['amount']:>10,.2f}){RESET}")
    print(f"  {'─' * 55}")


# ===========================================================================
# Core flows -- local broker
# ===========================================================================

def first_run_setup_local() -> dict:
    """Interactive first-run setup when no local state file exists."""
    default_tickers = TRADING_ASSISTANT_10
    print(f"\n  {BOLD}Welcome to Portfolio Trader!{RESET}")
    print(f"  {DIM}First-time setup -- let's configure your portfolio.{RESET}\n")

    # Starting cash
    while True:
        raw = input(f"  Starting cash (default ${DEFAULT_CASH:,.0f}): ").strip()
        if raw == "":
            cash = DEFAULT_CASH
            break
        try:
            cash = float(raw.replace(",", "").replace("$", ""))
            if cash <= 0:
                print(f"  {RED}Please enter a positive amount.{RESET}")
                continue
            break
        except ValueError:
            print(f"  {RED}Invalid number. Try again.{RESET}")

    # Tickers
    print(f"\n  Default tickers: {', '.join(default_tickers)}")
    choice = input(f"  Use defaults? (Y/n): ").strip().lower()
    if choice in ("n", "no"):
        raw = input(f"  Enter tickers (comma-separated): ").strip()
        tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
        if not tickers:
            print(f"  {YELLOW}No tickers entered, using defaults.{RESET}")
            tickers = default_tickers
    else:
        tickers = default_tickers

    state = create_local_initial_state(cash, tickers)
    save_state(state, LOCAL_STATE_FILE)
    print(f"\n  {GREEN}Portfolio created with ${cash:,.2f} and {len(tickers)} tickers.{RESET}")
    return state


def deposit_cash_local(state: dict) -> dict:
    """Add cash to the local portfolio."""
    while True:
        raw = input(f"\n  Amount to deposit: $").strip()
        try:
            amount = float(raw.replace(",", "").replace("$", ""))
            if amount <= 0:
                print(f"  {RED}Please enter a positive amount.{RESET}")
                continue
            break
        except ValueError:
            print(f"  {RED}Invalid number. Try again.{RESET}")

    state["cash"] += amount
    state["total_contributed"] = state.get("total_contributed", state["cash"]) + amount
    state["transactions"].append({
        "date": datetime.now().isoformat(timespec="seconds"),
        "action": "DEPOSIT",
        "ticker": "-",
        "shares": 0,
        "price": 0,
        "amount": amount,
    })
    save_state(state, LOCAL_STATE_FILE)
    print(f"  {GREEN}Deposited ${amount:,.2f}. New cash balance: ${state['cash']:,.2f}{RESET}\n")
    return state


def deposit_cash_alpaca(alpaca_broker: AlpacaBroker) -> None:
    """Guide user to add funds to their Alpaca paper account.

    Alpaca paper accounts can be reset to $100K from the dashboard.
    There is no API endpoint to deposit arbitrary amounts, so we
    show the current balance and point to the dashboard.
    """
    try:
        cash = alpaca_broker.get_cash()
        equity = alpaca_broker.get_equity()
        print(f"\n  {BOLD}Current Alpaca Paper Account{RESET}")
        print(f"  Cash:   ${cash:,.2f}")
        print(f"  Equity: ${equity:,.2f}")
    except Exception as e:
        print(f"\n  {RED}Could not fetch account info: {e}{RESET}")

    print(f"\n  {YELLOW}Alpaca paper accounts cannot receive deposits via API.{RESET}")
    print(f"  To add funds, reset your paper account from the Alpaca dashboard:")
    print(f"  {CYAN}{ALPACA_DASHBOARD_URL}{RESET}")
    print(f"  {DIM}(Account > Paper Trading > Reset Account){RESET}")
    print()


def get_trade_suggestions_local(state: dict) -> dict:
    """Fetch prices, run strategy, present and optionally execute trades (local)."""
    tickers = state["tickers"]
    prices, history = fetch_prices(tickers)

    if not prices:
        print(f"  {RED}Could not fetch any prices. Check your connection.{RESET}\n")
        return state

    # Auto-detect dividends
    state = check_dividends(state, history)

    # Build PortfolioState from saved state + live prices
    portfolio = PortfolioState(
        cash=state["cash"],
        positions={t: state["positions"].get(t, 0.0) for t in tickers},
        prices={t: prices.get(t, 0.0) for t in tickers},
    )

    # Compute optimal whole-share allocation and diff with current
    trades = compute_target_trades(portfolio, history)

    if not trades:
        print(f"  {GREEN}Portfolio is balanced -- no trades needed.{RESET}\n")
        return state

    # Present suggestions
    print(f"  {BOLD}SUGGESTED TRADES{RESET}")
    print(f"  {'─' * 55}")
    for i, trade in enumerate(trades):
        action = trade["action"]
        color = GREEN if action == "BUY" else RED
        print(f"  {color}{i + 1}. {action:<5} {trade['shares']:>5.0f} shares of "
              f"{trade['ticker']:<5} @ ${trade['price']:<9.2f} (${trade['amount']:>10,.2f}){RESET}")
    print(f"  {'─' * 55}")

    # Accept/reject
    print(f"  {CYAN}a{RESET} = accept all, "
          f"{CYAN}1-{len(trades)}{RESET} = accept individual (comma-separated), "
          f"{CYAN}n{RESET} = reject all")
    choice = input(f"  > ").strip().lower()

    if choice == "n" or choice == "":
        print(f"  {DIM}No trades executed.{RESET}\n")
        return state

    # Parse selection
    if choice == "a":
        selected = list(range(len(trades)))
    else:
        try:
            selected = [int(x.strip()) - 1 for x in choice.split(",")]
            selected = [s for s in selected if 0 <= s < len(trades)]
        except ValueError:
            print(f"  {RED}Invalid selection. No trades executed.{RESET}\n")
            return state

    if not selected:
        print(f"  {DIM}No valid trades selected.{RESET}\n")
        return state

    # Execute selected trades: sells first, then buys
    selected_trades = [trades[i] for i in selected]
    sells = [t for t in selected_trades if t["action"] == "SELL"]
    buys = [t for t in selected_trades if t["action"] == "BUY"]

    now = datetime.now().isoformat(timespec="seconds")
    cash = state["cash"]
    positions = dict(state["positions"])

    # Execute sells
    for trade in sells:
        t = trade["ticker"]
        shares = trade["shares"]
        amount = shares * trade["price"]
        positions[t] = positions.get(t, 0.0) - shares
        cash += amount
        state["transactions"].append({
            "date": now,
            "action": "SELL",
            "ticker": t,
            "shares": shares,
            "price": round(trade["price"], 2),
            "amount": round(amount, 2),
        })
        print(f"  {RED}SOLD {shares:.0f} shares of {t} for ${amount:,.2f}{RESET}")

    # Execute buys (with reserve enforcement, sequential)
    for trade in buys:
        t = trade["ticker"]
        price = trade["price"]
        # Recompute available cash with reserve
        total_eq = sum(positions.get(tk, 0.0) * prices.get(tk, 0.0)
                       for tk in tickers)
        total_val = cash + total_eq
        min_cash = total_val * CASH_RESERVE_PCT
        available = max(cash - min_cash, 0.0)
        # Floor to whole shares, cap at what simulation suggested
        max_shares = math.floor(available / price)
        buy_shares = min(trade["shares"], max_shares)
        if buy_shares < 1:
            print(f"  {YELLOW}Skipping {t} -- insufficient cash for 1 share.{RESET}")
            continue
        actual_spend = buy_shares * price
        positions[t] = positions.get(t, 0.0) + buy_shares
        cash -= actual_spend
        state["transactions"].append({
            "date": now,
            "action": "BUY",
            "ticker": t,
            "shares": buy_shares,
            "price": round(price, 2),
            "amount": round(actual_spend, 2),
        })
        print(f"  {GREEN}BOUGHT {buy_shares:.0f} shares of {t} for ${actual_spend:,.2f}{RESET}")

    state["cash"] = cash
    state["positions"] = positions
    save_state(state, LOCAL_STATE_FILE)
    print(f"\n  {GREEN}Trades executed and saved.{RESET}")
    print(f"  Cash remaining: ${cash:,.2f}\n")
    return state


def undo_last(state: dict) -> dict:
    """Undo the last transaction (or batch of transactions with the same timestamp)."""
    txns = state.get("transactions", [])
    if len(txns) <= 1:
        print(f"\n  {YELLOW}Nothing to undo (only the initial deposit remains).{RESET}\n")
        return state

    # Find the timestamp of the last transaction and undo all with that timestamp
    last_ts = txns[-1]["date"]
    to_undo = []
    while txns and txns[-1]["date"] == last_ts:
        to_undo.append(txns.pop())

    # Show what we're undoing
    print(f"\n  {BOLD}Undoing {len(to_undo)} transaction(s) from {last_ts}:{RESET}")
    for tx in to_undo:
        action = tx["action"]
        if action == "BUY":
            state["positions"][tx["ticker"]] = state["positions"].get(tx["ticker"], 0) - tx["shares"]
            state["cash"] += tx["amount"]
            print(f"  {DIM}  Reversed BUY {tx['shares']:.0f} x {tx['ticker']}{RESET}")
        elif action == "SELL":
            state["positions"][tx["ticker"]] = state["positions"].get(tx["ticker"], 0) + tx["shares"]
            state["cash"] -= tx["amount"]
            print(f"  {DIM}  Reversed SELL {tx['shares']:.0f} x {tx['ticker']}{RESET}")
        elif action == "DEPOSIT":
            state["cash"] -= tx["amount"]
            state["total_contributed"] = state.get("total_contributed", 0) - tx["amount"]
            print(f"  {DIM}  Reversed DEPOSIT ${tx['amount']:,.2f}{RESET}")
        elif action == "DIVIDEND":
            state["cash"] -= tx["amount"]
            print(f"  {DIM}  Reversed DIVIDEND ${tx['amount']:,.2f} from {tx['ticker']}{RESET}")

    save_state(state, LOCAL_STATE_FILE)
    print(f"  {GREEN}Done. Cash: ${state['cash']:,.2f}{RESET}\n")
    return state


# ===========================================================================
# Alpaca suggestion engine
# ===========================================================================

def compute_alpaca_suggestions(positions: dict[str, int],
                               prices: dict[str, float],
                               cash: float,
                               history: dict[str, pd.DataFrame]
                               ) -> list[dict]:
    """Compute inv-vol EW target trades for Alpaca.

    Reuses compute_target_shares + compute_rebalance_trades from equal_weight.

    Returns list of trade dicts: {action, ticker, shares, price, amount}.
    """
    total_equity = sum(positions.get(t, 0) * prices.get(t, 0)
                       for t in positions)
    total_value = cash + total_equity
    if total_value <= 0:
        return []

    target_shares = compute_target_shares(
        positions=positions,
        prices=prices,
        cash=cash,
        cash_reserve_pct=CASH_RESERVE_PCT,
        history=history,
    )

    return compute_rebalance_trades(
        current_positions=positions,
        target_shares=target_shares,
        prices=prices,
        total_value=total_value,
        cash_reserve_pct=CASH_RESERVE_PCT,
        cash=cash,
    )


# ===========================================================================
# Main loop -- local broker
# ===========================================================================

def run_local():
    """Entry point for local broker interactive CLI loop."""
    print_header("Local (JSON)")

    # Load or create state
    if os.path.exists(LOCAL_STATE_FILE):
        state = load_state(LOCAL_STATE_FILE)
        print(f"  {GREEN}Loaded portfolio with {len(state['tickers'])} tickers "
              f"and ${state['cash']:,.2f} cash.{RESET}\n")
    else:
        state = first_run_setup_local()
        print()

    # Cache prices, risk-free rate history, and Sharpe stats for the session
    cached_prices = None
    cached_sharpe = None
    cached_rf_history = None

    while True:
        print_local_menu()
        choice = input(f"  {BOLD}>{RESET} ").strip()

        if choice == "1":
            if cached_prices is None:
                cached_prices, _ = fetch_prices(state["tickers"])
            # Lazily compute Sharpe on first view
            if cached_sharpe is None and cached_prices:
                txns = state.get("transactions", [])
                if len(txns) > 0:
                    start = txns[0]["date"][:10]
                    if cached_rf_history is None:
                        cached_rf_history = fetch_risk_free_history(start)
                    full_hist = fetch_full_history(state["tickers"], start)
                    cached_sharpe = compute_portfolio_stats(
                        txns, state["tickers"], full_hist,
                        rf_history=cached_rf_history)
            view_portfolio_local(state, cached_prices, cached_sharpe)

        elif choice == "2":
            state = get_trade_suggestions_local(state)
            cached_prices = None
            cached_sharpe = None
            cached_rf_history = None

        elif choice == "3":
            state = deposit_cash_local(state)
            cached_sharpe = None
            cached_rf_history = None

        elif choice == "4":
            if cached_prices is None:
                cached_prices, _ = fetch_prices(state["tickers"])
            state = sell_shares_local(state, cached_prices)
            cached_prices = None
            cached_sharpe = None
            cached_rf_history = None

        elif choice == "5":
            show_transaction_history(state)

        elif choice == "6":
            if cached_prices is None:
                cached_prices, _ = fetch_prices(state["tickers"])
            view_gains(state, cached_prices)

        elif choice == "7":
            cached_prices, _ = fetch_prices(state["tickers"])
            cached_sharpe = None
            cached_rf_history = None
            # Recompute Sharpe after refresh
            txns = state.get("transactions", [])
            if len(txns) > 0:
                start = txns[0]["date"][:10]
                cached_rf_history = fetch_risk_free_history(start)
                full_hist = fetch_full_history(state["tickers"], start)
                cached_sharpe = compute_portfolio_stats(
                    txns, state["tickers"], full_hist,
                    rf_history=cached_rf_history)
            view_portfolio_local(state, cached_prices, cached_sharpe)

        elif choice == "8":
            state = undo_last(state)
            cached_prices = None
            cached_sharpe = None
            cached_rf_history = None

        elif choice == "9":
            save_state(state, LOCAL_STATE_FILE)
            print(f"\n  {GREEN}Portfolio saved. Goodbye!{RESET}\n")
            break

        else:
            print(f"  {YELLOW}Please enter 1-9.{RESET}\n")


# ===========================================================================
# Main loop -- Alpaca broker
# ===========================================================================

def run_alpaca(profile: str):
    """Entry point for Alpaca broker interactive CLI loop."""
    print_header(f"Alpaca ({profile})")

    print(f"  {DIM}Config: ~/.alpaca/config.yaml  |  Profile: {profile}{RESET}")

    # Initialize broker
    try:
        alpaca_broker = AlpacaBroker(profile=profile)
        print(f"  {GREEN}Connected to Alpaca paper trading.{RESET}")
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"  {RED}{e}{RESET}")
        print(f"  {DIM}Configure ~/.alpaca/config.yaml or set "
              f"ALPACA_API_KEY / ALPACA_SECRET_KEY env vars.{RESET}")
        return

    # Show market status
    try:
        if alpaca_broker.is_market_open():
            print(f"  {GREEN}Market is OPEN.{RESET}")
        else:
            print(f"  {YELLOW}Market is CLOSED. Orders will queue for next open.{RESET}")
    except Exception:
        print(f"  {DIM}Could not check market status.{RESET}")

    # Validate tickers against Alpaca's tradeable assets
    default_tickers = list(ALL_TICKERS)
    print(f"\n{CYAN}  Checking tradeable assets...{RESET}")
    try:
        _, excluded = alpaca_broker.filter_tradeable(default_tickers)
        if excluded:
            print(f"  {YELLOW}Not tradeable on Alpaca: "
                  f"{', '.join(excluded)}{RESET}")
            print(f"  {DIM}These tickers will be excluded from "
                  f"suggestions.{RESET}")
    except Exception as e:
        excluded = []
        print(f"  {DIM}Could not validate assets: {e}{RESET}")

    # Load or create local state
    if os.path.exists(ALPACA_STATE_FILE):
        state = load_state(ALPACA_STATE_FILE)
        # Remove any excluded tickers from loaded state
        if excluded:
            before = len(state["tickers"])
            state["tickers"] = [t for t in state["tickers"]
                                if t not in excluded]
            if len(state["tickers"]) < before:
                save_state(state, ALPACA_STATE_FILE)
        print(f"  Loaded session with {len(state['tickers'])} tickers.\n")
    else:
        # First run -- choose tickers
        print(f"\n  {BOLD}First-time setup{RESET}")
        valid_defaults = [t for t in default_tickers if t not in excluded]
        print(f"  Default tickers: {len(valid_defaults)} "
              f"(from {len(default_tickers)} universe, "
              f"{len(excluded)} excluded)")
        choice = input(f"  Use defaults? (Y/n): ").strip().lower()
        if choice in ("n", "no"):
            raw = input(f"  Enter tickers (comma-separated): ").strip()
            tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
            if not tickers:
                print(f"  {YELLOW}No tickers entered, using defaults.{RESET}")
                tickers = valid_defaults
        else:
            tickers = valid_defaults
        state = create_alpaca_initial_state(tickers)
        save_state(state, ALPACA_STATE_FILE)
        print(f"  {GREEN}Session created with {len(tickers)} tickers.{RESET}\n")

    # Cache for current session
    cached_positions = None
    cached_prices = None
    cached_cash = None
    cached_pos_info = None
    cached_history = None
    cached_trades = None

    while True:
        print_alpaca_menu()
        choice = input(f"  {BOLD}>{RESET} ").strip()

        if choice == "1":
            # View Portfolio
            print(f"\n{CYAN}  Fetching from Alpaca...{RESET}")
            try:
                alpaca_positions = alpaca_broker.get_positions()
                cash = alpaca_broker.get_cash()
                positions = {}
                prices = {}
                for t in state["tickers"]:
                    if t in alpaca_positions:
                        pos = alpaca_positions[t]
                        positions[t] = pos.qty
                        prices[t] = pos.current_price
                    else:
                        positions[t] = 0
                        try:
                            ticker = yf.Ticker(t)
                            hist = ticker.history(period="5d")
                            if len(hist) > 0:
                                prices[t] = float(hist["Close"].iloc[-1])
                            else:
                                prices[t] = 0.0
                        except Exception:
                            prices[t] = 0.0
                cached_positions = positions
                cached_prices = prices
                cached_cash = cash
                cached_pos_info = alpaca_positions
                state["last_refresh"] = datetime.now().isoformat(timespec="seconds")
                display_portfolio_alpaca(positions, prices, cash,
                                         state["tickers"], alpaca_positions)
            except Exception as e:
                print(f"  {RED}Error: {e}{RESET}\n")

        elif choice == "2":
            # Get Suggestions
            print(f"\n{CYAN}  Fetching data...{RESET}")
            try:
                alpaca_positions = alpaca_broker.get_positions()
                cash = alpaca_broker.get_cash()
                positions = {}
                prices = {}
                for t in state["tickers"]:
                    if t in alpaca_positions:
                        pos = alpaca_positions[t]
                        positions[t] = pos.qty
                        prices[t] = pos.current_price
                    else:
                        positions[t] = 0
                        try:
                            ticker = yf.Ticker(t)
                            hist = ticker.history(period="5d")
                            if len(hist) > 0:
                                prices[t] = float(hist["Close"].iloc[-1])
                            else:
                                prices[t] = 0.0
                        except Exception:
                            prices[t] = 0.0
                cached_positions = positions
                cached_prices = prices
                cached_cash = cash
                cached_pos_info = alpaca_positions
                history = fetch_history(state["tickers"])
                cached_history = history

                trades = compute_alpaca_suggestions(positions, prices, cash, history)
                cached_trades = trades
                display_suggestions(trades)

                if trades:
                    print(f"  Use option {CYAN}3{RESET} to execute these trades.\n")
            except Exception as e:
                print(f"  {RED}Error: {e}{RESET}\n")

        elif choice == "3":
            # Execute Trades
            if not cached_trades:
                print(f"  {YELLOW}No suggestions to execute. "
                      f"Run option 2 first.{RESET}\n")
                continue

            display_suggestions(cached_trades)
            print(f"  {CYAN}a{RESET} = accept all, "
                  f"{CYAN}1-{len(cached_trades)}{RESET} = select (comma-separated), "
                  f"{CYAN}n{RESET} = cancel")
            sel = input(f"  > ").strip().lower()

            if sel == "n" or sel == "":
                print(f"  {DIM}Cancelled.{RESET}\n")
                continue

            if sel == "a":
                selected_trades = cached_trades
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in sel.split(",")]
                    selected_trades = [cached_trades[i] for i in indices
                                       if 0 <= i < len(cached_trades)]
                except (ValueError, IndexError):
                    print(f"  {RED}Invalid selection.{RESET}\n")
                    continue

            if not selected_trades:
                print(f"  {DIM}No valid trades selected.{RESET}\n")
                continue

            print()
            results = execute_alpaca_trades(alpaca_broker, selected_trades, state)
            save_state(state, ALPACA_STATE_FILE)

            filled = sum(1 for r in results if r.status == "filled")
            total_attempted = len(selected_trades)
            print(f"\n  {GREEN}{filled}/{total_attempted} orders filled.{RESET}")

            # Show failed/errored orders
            failed = [r for r in results if r.status != "filled"]
            if failed:
                print(f"  {YELLOW}Failed orders:{RESET}")
                for r in failed:
                    print(f"    {RED}{r.ticker}: {r.status}{RESET}")
            # Show errors from exception path
            recent_txns = state["transactions"][-total_attempted:]
            errored = [tx for tx in recent_txns if tx.get("status") == "error"]
            if errored:
                if not failed:
                    print(f"  {YELLOW}Failed orders:{RESET}")
                for tx in errored:
                    err_msg = tx.get("error", "unknown error")
                    print(f"    {RED}{tx['ticker']}: {err_msg}{RESET}")

            # Invalidate caches
            cached_positions = None
            cached_prices = None
            cached_cash = None
            cached_pos_info = None
            cached_trades = None
            print()

        elif choice == "4":
            # Sell Shares
            if cached_positions is None or cached_prices is None:
                print(f"\n{CYAN}  Fetching from Alpaca...{RESET}")
                try:
                    alpaca_positions = alpaca_broker.get_positions()
                    cash = alpaca_broker.get_cash()
                    positions = {}
                    prices = {}
                    for t in state["tickers"]:
                        if t in alpaca_positions:
                            pos = alpaca_positions[t]
                            positions[t] = pos.qty
                            prices[t] = pos.current_price
                        else:
                            positions[t] = 0
                            try:
                                ticker = yf.Ticker(t)
                                hist = ticker.history(period="5d")
                                if len(hist) > 0:
                                    prices[t] = float(hist["Close"].iloc[-1])
                                else:
                                    prices[t] = 0.0
                            except Exception:
                                prices[t] = 0.0
                    cached_positions = positions
                    cached_prices = prices
                    cached_cash = cash
                    cached_pos_info = alpaca_positions
                except Exception as e:
                    print(f"  {RED}Error: {e}{RESET}\n")
                    continue

            traded = sell_shares_alpaca(
                alpaca_broker, state, cached_positions, cached_prices)
            if traded:
                cached_positions = None
                cached_prices = None
                cached_cash = None
                cached_pos_info = None
                cached_trades = None

        elif choice == "5":
            # Deposit Cash
            deposit_cash_alpaca(alpaca_broker)

        elif choice == "6":
            # Refresh Prices
            print(f"\n{CYAN}  Fetching from Alpaca...{RESET}")
            try:
                alpaca_positions = alpaca_broker.get_positions()
                cash = alpaca_broker.get_cash()
                positions = {}
                prices = {}
                for t in state["tickers"]:
                    if t in alpaca_positions:
                        pos = alpaca_positions[t]
                        positions[t] = pos.qty
                        prices[t] = pos.current_price
                    else:
                        positions[t] = 0
                        try:
                            ticker = yf.Ticker(t)
                            hist = ticker.history(period="5d")
                            if len(hist) > 0:
                                prices[t] = float(hist["Close"].iloc[-1])
                            else:
                                prices[t] = 0.0
                        except Exception:
                            prices[t] = 0.0
                cached_positions = positions
                cached_prices = prices
                cached_cash = cash
                cached_pos_info = alpaca_positions
                cached_trades = None
                state["last_refresh"] = datetime.now().isoformat(timespec="seconds")
                display_portfolio_alpaca(positions, prices, cash,
                                         state["tickers"], alpaca_positions)
            except Exception as e:
                print(f"  {RED}Error: {e}{RESET}\n")

        elif choice == "7":
            # Transaction Log
            show_alpaca_transaction_log(state)

        elif choice == "8":
            # Market Status
            try:
                clock = alpaca_broker._get_client().get_clock()
                if clock.is_open:
                    print(f"\n  {GREEN}Market is OPEN.{RESET}")
                    print(f"  Closes at: {clock.next_close}")
                else:
                    print(f"\n  {YELLOW}Market is CLOSED.{RESET}")
                    print(f"  Next open: {clock.next_open}")
                print()
            except Exception as e:
                print(f"  {RED}Error: {e}{RESET}\n")

        elif choice == "9":
            # Save & Exit
            save_state(state, ALPACA_STATE_FILE)
            print(f"\n  {GREEN}Session saved. Goodbye!{RESET}\n")
            break

        else:
            print(f"  {YELLOW}Please enter 1-9.{RESET}\n")


# ===========================================================================
# CLI entry point
# ===========================================================================

DEFAULT_TICKERS = TRADING_ASSISTANT_10


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Portfolio Trader -- Inverse-Volatility Equal Weight")
    parser.add_argument(
        "--broker", choices=["local", "alpaca"], default="local",
        help="Broker backend: local (JSON state) or alpaca (API). Default: local")
    parser.add_argument(
        "--profile", default="portfolio",
        help="Alpaca config profile (only used with --broker alpaca). Default: portfolio")
    return parser.parse_args()


def main():
    """Entry point -- dispatch to the appropriate broker loop."""
    args = parse_args()

    if args.broker == "alpaca":
        run_alpaca(args.profile)
    else:
        run_local()


if __name__ == "__main__":
    main()
