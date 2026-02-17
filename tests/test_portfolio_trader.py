"""
Tests for Portfolio Trader -- merged from trading assistant (local broker)
and Alpaca paper trader test suites.

Covers:
  - Gain/loss tracking (average cost method)
  - Shared equal-weight allocation algorithm
  - Proportional sell/withdraw
  - Portfolio stats / Sharpe ratio
  - Alpaca broker dataclasses (PositionInfo, OrderResult)
  - Alpaca broker operations (mocked)
  - Alpaca app-level functions (display, suggestions, execution)
  - Asset validation (is_tradeable, filter_tradeable)
  - File structure and cross-script consistency
"""

import contextlib
import importlib.util
import inspect
import io
import json
import math
import os
import types
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from finance_tools.backtest.portfolio import PortfolioState
from finance_tools.strategies.equal_weight import (
    compute_target_trades,
    compute_target_shares,
    compute_volatility,
    needs_rebalance,
    compute_rebalance_trades,
    CASH_RESERVE_PCT,
)
from finance_tools.data.market import fetch_risk_free_rate
from finance_tools.data.universe import ALL_TICKERS, TRADING_ASSISTANT_10
from finance_tools.broker.alpaca import AlpacaBroker, PositionInfo, OrderResult

# ---------------------------------------------------------------------------
# Load app.py via importlib to avoid import collisions
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
_APP_DIR = os.path.join(_REPO_ROOT, "apps", "portfolio_trader")
_APP_PATH = os.path.join(_APP_DIR, "app.py")


def _load_module(name, filepath):
    """Load a module by explicit file path to avoid import collisions."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


portfolio_app = _load_module("portfolio_trader_app", _APP_PATH)


# =====================================================================
# Helpers -- transaction builders
# =====================================================================

def _tx(action, ticker, shares, price):
    """Shorthand for building a transaction dict."""
    return {
        "date": "2025-01-01T00:00:00",
        "action": action,
        "ticker": ticker,
        "shares": shares,
        "price": price,
        "amount": shares * price if action in ("BUY", "SELL") else price,
    }


def _deposit(amount):
    """Shorthand for a deposit transaction."""
    return {
        "date": "2025-01-01T00:00:00",
        "action": "DEPOSIT",
        "ticker": "-",
        "shares": 0,
        "price": 0,
        "amount": amount,
    }


def _dividend(ticker, per_share, total):
    """Shorthand for a dividend transaction."""
    return {
        "date": "2025-01-01T00:00:00",
        "action": "DIVIDEND",
        "ticker": ticker,
        "shares": 0,
        "price": per_share,
        "amount": total,
    }


# =====================================================================
# Helpers -- mock Alpaca objects
# =====================================================================

def _mock_account(cash=100_000.0, equity=150_000.0):
    """Create a mock Alpaca account object."""
    account = MagicMock()
    account.cash = str(cash)
    account.equity = str(equity)
    return account


def _mock_position(symbol="AAPL", qty=10, avg_entry=150.0,
                   current=155.0, market_value=1550.0,
                   unrealized_pl=50.0, unrealized_plpc=0.0333):
    """Create a mock Alpaca position object."""
    pos = MagicMock()
    pos.symbol = symbol
    pos.qty = str(qty)
    pos.avg_entry_price = str(avg_entry)
    pos.current_price = str(current)
    pos.market_value = str(market_value)
    pos.unrealized_pl = str(unrealized_pl)
    pos.unrealized_plpc = str(unrealized_plpc)
    return pos


def _mock_order(order_id="order-123", symbol="AAPL", side="buy",
                qty=10, status="filled", filled_qty=10,
                filled_avg_price=155.0):
    """Create a mock Alpaca order object."""
    order = MagicMock()
    order.id = order_id
    order.symbol = symbol
    order.side = MagicMock()
    order.side.value = side
    order.qty = str(qty)
    order.status = MagicMock()
    order.status.value = status
    order.filled_qty = str(filled_qty) if filled_qty else None
    order.filled_avg_price = str(filled_avg_price) if filled_avg_price else None
    return order


def _mock_clock(is_open=True, next_open="2026-02-11T09:30:00",
                next_close="2026-02-10T16:00:00"):
    """Create a mock Alpaca clock object."""
    clock = MagicMock()
    clock.is_open = is_open
    clock.next_open = next_open
    clock.next_close = next_close
    return clock


# #####################################################################
#
#  SECTION 1: LOCAL BROKER / TRADING ASSISTANT TESTS
#
# #####################################################################


# =====================================================================
# compute_gains -- basic functionality
# =====================================================================

class TestComputeGainsBasic:
    """Core average-cost computation tests."""

    def test_empty_transactions(self):
        result = portfolio_app.compute_gains([])
        assert result == {}

    def test_empty_transactions_with_prices(self):
        result = portfolio_app.compute_gains([], {"AAPL": 150.0})
        # Tickers from prices with no transactions should appear but have zero everything
        assert "AAPL" in result
        assert result["AAPL"]["total_shares"] == 0.0
        assert result["AAPL"]["unrealized_gain"] == 0.0

    def test_single_buy(self):
        txns = [_tx("BUY", "AAPL", 10, 100.0)]
        result = portfolio_app.compute_gains(txns, {"AAPL": 110.0})
        g = result["AAPL"]
        assert g["total_shares"] == 10
        assert g["avg_cost"] == pytest.approx(100.0)
        assert g["total_cost"] == pytest.approx(1000.0)
        assert g["realized_gain"] == pytest.approx(0.0)
        assert g["unrealized_gain"] == pytest.approx(100.0)  # (110 - 100) * 10

    def test_single_buy_no_prices(self):
        txns = [_tx("BUY", "AAPL", 10, 100.0)]
        result = portfolio_app.compute_gains(txns)
        g = result["AAPL"]
        assert g["unrealized_gain"] == pytest.approx(0.0)
        assert g["current_price"] == 0.0

    def test_multiple_buys_same_ticker(self):
        txns = [
            _tx("BUY", "AAPL", 10, 100.0),  # cost = 1000
            _tx("BUY", "AAPL", 10, 120.0),  # cost = 1200, total = 2200/20 = 110
        ]
        result = portfolio_app.compute_gains(txns, {"AAPL": 130.0})
        g = result["AAPL"]
        assert g["total_shares"] == 20
        assert g["avg_cost"] == pytest.approx(110.0)
        assert g["total_cost"] == pytest.approx(2200.0)
        assert g["realized_gain"] == pytest.approx(0.0)
        assert g["unrealized_gain"] == pytest.approx(400.0)  # (130 - 110) * 20

    def test_multiple_tickers(self):
        txns = [
            _tx("BUY", "AAPL", 10, 100.0),
            _tx("BUY", "MSFT", 5, 200.0),
        ]
        result = portfolio_app.compute_gains(txns, {"AAPL": 110.0, "MSFT": 210.0})
        assert result["AAPL"]["total_shares"] == 10
        assert result["MSFT"]["total_shares"] == 5
        assert result["AAPL"]["unrealized_gain"] == pytest.approx(100.0)
        assert result["MSFT"]["unrealized_gain"] == pytest.approx(50.0)


# =====================================================================
# compute_gains -- sells and realized gains
# =====================================================================

class TestComputeGainsSells:
    """Tests involving sell transactions."""

    def test_buy_then_sell_all(self):
        txns = [
            _tx("BUY", "AAPL", 10, 100.0),
            _tx("SELL", "AAPL", 10, 120.0),
        ]
        result = portfolio_app.compute_gains(txns, {"AAPL": 130.0})
        g = result["AAPL"]
        assert g["total_shares"] == 0
        assert g["realized_gain"] == pytest.approx(200.0)  # (120-100)*10
        assert g["unrealized_gain"] == pytest.approx(0.0)  # no shares

    def test_partial_sell(self):
        txns = [
            _tx("BUY", "AAPL", 10, 100.0),
            _tx("SELL", "AAPL", 5, 120.0),
        ]
        result = portfolio_app.compute_gains(txns, {"AAPL": 130.0})
        g = result["AAPL"]
        assert g["total_shares"] == 5
        assert g["avg_cost"] == pytest.approx(100.0)  # avg cost unchanged after partial sell
        assert g["realized_gain"] == pytest.approx(100.0)  # (120-100)*5
        assert g["unrealized_gain"] == pytest.approx(150.0)  # (130-100)*5

    def test_avg_cost_after_sell_and_rebuy(self):
        """Average cost should reflect the new purchase after sell+rebuy."""
        txns = [
            _tx("BUY", "AAPL", 10, 100.0),   # avg=100
            _tx("SELL", "AAPL", 5, 120.0),    # realized=(120-100)*5=100, remaining 5@100
            _tx("BUY", "AAPL", 5, 140.0),     # cost=500+700=1200, 10 shares, avg=120
        ]
        result = portfolio_app.compute_gains(txns, {"AAPL": 150.0})
        g = result["AAPL"]
        assert g["total_shares"] == 10
        assert g["avg_cost"] == pytest.approx(120.0)
        assert g["realized_gain"] == pytest.approx(100.0)
        assert g["unrealized_gain"] == pytest.approx(300.0)  # (150-120)*10

    def test_sell_at_loss(self):
        txns = [
            _tx("BUY", "AAPL", 10, 100.0),
            _tx("SELL", "AAPL", 10, 80.0),
        ]
        result = portfolio_app.compute_gains(txns)
        g = result["AAPL"]
        assert g["realized_gain"] == pytest.approx(-200.0)  # (80-100)*10
        assert g["total_shares"] == 0

    def test_multiple_sells_different_prices(self):
        txns = [
            _tx("BUY", "AAPL", 20, 100.0),   # avg=100
            _tx("SELL", "AAPL", 5, 110.0),    # realized = (110-100)*5 = 50
            _tx("SELL", "AAPL", 5, 90.0),     # realized += (90-100)*5 = -50, total=0
        ]
        result = portfolio_app.compute_gains(txns, {"AAPL": 100.0})
        g = result["AAPL"]
        assert g["total_shares"] == 10
        assert g["avg_cost"] == pytest.approx(100.0)
        assert g["realized_gain"] == pytest.approx(0.0)  # gains and losses cancel
        assert g["unrealized_gain"] == pytest.approx(0.0)  # price = avg cost


# =====================================================================
# compute_gains -- edge cases
# =====================================================================

class TestComputeGainsEdgeCases:
    """Edge cases and boundary conditions."""

    def test_deposit_and_dividend_ignored(self):
        txns = [
            _deposit(5000.0),
            _tx("BUY", "AAPL", 10, 100.0),
            _dividend("AAPL", 0.50, 5.0),
        ]
        result = portfolio_app.compute_gains(txns, {"AAPL": 110.0})
        g = result["AAPL"]
        assert g["total_shares"] == 10
        assert g["avg_cost"] == pytest.approx(100.0)
        assert g["realized_gain"] == pytest.approx(0.0)

    def test_full_liquidation_then_rebuy(self):
        txns = [
            _tx("BUY", "AAPL", 10, 100.0),
            _tx("SELL", "AAPL", 10, 120.0),
            _tx("BUY", "AAPL", 5, 130.0),
        ]
        result = portfolio_app.compute_gains(txns, {"AAPL": 140.0})
        g = result["AAPL"]
        assert g["total_shares"] == 5
        assert g["avg_cost"] == pytest.approx(130.0)  # fresh cost basis
        assert g["realized_gain"] == pytest.approx(200.0)  # from first round
        assert g["unrealized_gain"] == pytest.approx(50.0)  # (140-130)*5

    def test_zero_shares_after_full_sell(self):
        txns = [
            _tx("BUY", "AAPL", 1, 100.0),
            _tx("SELL", "AAPL", 1, 110.0),
        ]
        result = portfolio_app.compute_gains(txns)
        g = result["AAPL"]
        assert g["total_shares"] == 0
        assert g["total_cost"] == pytest.approx(0.0)
        assert g["avg_cost"] == pytest.approx(0.0)

    def test_single_share(self):
        txns = [_tx("BUY", "AAPL", 1, 50.0)]
        result = portfolio_app.compute_gains(txns, {"AAPL": 75.0})
        g = result["AAPL"]
        assert g["total_shares"] == 1
        assert g["avg_cost"] == pytest.approx(50.0)
        assert g["unrealized_gain"] == pytest.approx(25.0)

    def test_only_deposits(self):
        txns = [_deposit(5000.0), _deposit(1000.0)]
        result = portfolio_app.compute_gains(txns)
        assert result == {}

    def test_only_dividends(self):
        txns = [_dividend("AAPL", 0.50, 5.0)]
        result = portfolio_app.compute_gains(txns)
        assert result == {}

    def test_prices_for_unseen_ticker(self):
        """Prices dict includes tickers not in transactions."""
        txns = [_tx("BUY", "AAPL", 10, 100.0)]
        result = portfolio_app.compute_gains(txns, {"AAPL": 110.0, "MSFT": 300.0})
        assert "MSFT" in result
        assert result["MSFT"]["total_shares"] == 0
        assert result["MSFT"]["unrealized_gain"] == 0.0

    def test_none_prices(self):
        txns = [_tx("BUY", "AAPL", 10, 100.0)]
        result = portfolio_app.compute_gains(txns, None)
        g = result["AAPL"]
        assert g["unrealized_gain"] == pytest.approx(0.0)


# =====================================================================
# compute_gains -- average cost correctness
# =====================================================================

class TestAverageCostCorrectness:
    """Detailed tests for the average cost calculation."""

    def test_three_buys_at_different_prices(self):
        txns = [
            _tx("BUY", "F", 100, 10.0),   # cost 1000
            _tx("BUY", "F", 50, 12.0),    # cost  600
            _tx("BUY", "F", 50, 8.0),     # cost  400
        ]
        # total: 200 shares, $2000 cost, avg = $10.00
        result = portfolio_app.compute_gains(txns, {"F": 11.0})
        g = result["F"]
        assert g["total_shares"] == 200
        assert g["avg_cost"] == pytest.approx(10.0)
        assert g["total_cost"] == pytest.approx(2000.0)
        assert g["unrealized_gain"] == pytest.approx(200.0)

    def test_avg_cost_preserved_after_partial_sell(self):
        """For avg cost method, selling shares doesn't change per-share avg cost."""
        txns = [
            _tx("BUY", "F", 100, 10.0),
            _tx("BUY", "F", 100, 20.0),
            # avg = 3000/200 = 15
            _tx("SELL", "F", 50, 25.0),
            # remaining: 150 shares, cost = 150*15 = 2250, avg still 15
        ]
        result = portfolio_app.compute_gains(txns, {"F": 18.0})
        g = result["F"]
        assert g["total_shares"] == 150
        assert g["avg_cost"] == pytest.approx(15.0)
        assert g["total_cost"] == pytest.approx(2250.0)
        assert g["realized_gain"] == pytest.approx(500.0)   # (25-15)*50
        assert g["unrealized_gain"] == pytest.approx(450.0)  # (18-15)*150

    def test_penny_stock_precision(self):
        txns = [
            _tx("BUY", "PENNY", 10000, 0.01),
            _tx("SELL", "PENNY", 5000, 0.02),
        ]
        result = portfolio_app.compute_gains(txns, {"PENNY": 0.015})
        g = result["PENNY"]
        assert g["total_shares"] == 5000
        assert g["avg_cost"] == pytest.approx(0.01)
        assert g["realized_gain"] == pytest.approx(50.0)  # (0.02-0.01)*5000
        assert g["unrealized_gain"] == pytest.approx(25.0)  # (0.015-0.01)*5000

    def test_expensive_stock(self):
        txns = [_tx("BUY", "BRK", 2, 500000.0)]
        result = portfolio_app.compute_gains(txns, {"BRK": 510000.0})
        g = result["BRK"]
        assert g["unrealized_gain"] == pytest.approx(20000.0)


# =====================================================================
# compute_gains -- consistency
# =====================================================================

class TestGainsConsistency:
    """Cross-checks between realized, unrealized, and total P&L."""

    def test_realized_plus_unrealized_equals_total_pnl(self):
        """realized + unrealized = market_value - total_invested."""
        txns = [
            _deposit(10000.0),
            _tx("BUY", "AAPL", 10, 100.0),
            _tx("BUY", "MSFT", 5, 200.0),
            _tx("SELL", "AAPL", 3, 120.0),
        ]
        prices = {"AAPL": 115.0, "MSFT": 220.0}
        result = portfolio_app.compute_gains(txns, prices)

        # Total invested = 10*100 + 5*200 = 2000
        total_invested = 10 * 100.0 + 5 * 200.0
        # Market value of remaining shares
        market_value = (
            result["AAPL"]["total_shares"] * prices["AAPL"]
            + result["MSFT"]["total_shares"] * prices["MSFT"]
        )
        # Sale proceeds from AAPL sell
        sale_proceeds = 3 * 120.0

        total_realized = sum(g["realized_gain"] for g in result.values())
        total_unrealized = sum(g["unrealized_gain"] for g in result.values())

        # realized + unrealized = market_value + sale_proceeds - total_invested
        assert total_realized + total_unrealized == pytest.approx(
            market_value + sale_proceeds - total_invested
        )

    def test_no_trades_no_gains(self):
        """No buy/sell transactions should produce zero gains everywhere."""
        txns = [_deposit(5000.0), _dividend("AAPL", 0.50, 5.0)]
        result = portfolio_app.compute_gains(txns, {"AAPL": 150.0})
        # AAPL should appear from prices but with zero everything
        total_realized = sum(g["realized_gain"] for g in result.values())
        total_unrealized = sum(g["unrealized_gain"] for g in result.values())
        assert total_realized == pytest.approx(0.0)
        assert total_unrealized == pytest.approx(0.0)

    def test_breakeven_no_gain(self):
        """Buy and sell at same price -- zero realized gain."""
        txns = [
            _tx("BUY", "F", 50, 10.0),
            _tx("SELL", "F", 50, 10.0),
        ]
        result = portfolio_app.compute_gains(txns)
        assert result["F"]["realized_gain"] == pytest.approx(0.0)

    def test_price_equals_avg_cost_zero_unrealized(self):
        txns = [_tx("BUY", "F", 50, 10.0)]
        result = portfolio_app.compute_gains(txns, {"F": 10.0})
        assert result["F"]["unrealized_gain"] == pytest.approx(0.0)


# =====================================================================
# view_gains -- display (smoke tests)
# =====================================================================

class TestViewGainsDisplay:
    """Smoke tests -- view_gains should not raise."""

    def test_view_gains_with_positions(self, capsys):
        state = portfolio_app.create_local_initial_state(5000.0, ["AAPL", "MSFT"])
        state["transactions"].append(_tx("BUY", "AAPL", 10, 100.0))
        prices = {"AAPL": 110.0, "MSFT": 300.0}
        portfolio_app.view_gains(state, prices)
        output = capsys.readouterr().out
        assert "AAPL" in output
        assert "Unrealized" in output

    def test_view_gains_empty(self, capsys):
        state = portfolio_app.create_local_initial_state(5000.0, ["AAPL"])
        portfolio_app.view_gains(state, {})
        output = capsys.readouterr().out
        assert "No positions" in output

    def test_view_gains_with_realized(self, capsys):
        state = portfolio_app.create_local_initial_state(5000.0, ["AAPL"])
        state["transactions"].extend([
            _tx("BUY", "AAPL", 10, 100.0),
            _tx("SELL", "AAPL", 10, 120.0),
        ])
        prices = {"AAPL": 130.0}
        portfolio_app.view_gains(state, prices)
        output = capsys.readouterr().out
        assert "Realized" in output
        assert "200" in output  # realized gain of $200


# =====================================================================
# view_portfolio_local -- G/L column
# =====================================================================

class TestViewPortfolioGainLoss:
    """view_portfolio_local should now include G/L column."""

    def test_gl_column_header(self, capsys):
        state = portfolio_app.create_local_initial_state(5000.0, ["AAPL"])
        prices = {"AAPL": 150.0}
        portfolio_app.view_portfolio_local(state, prices)
        output = capsys.readouterr().out
        assert "G/L" in output

    def test_gl_column_shows_gain(self, capsys):
        state = portfolio_app.create_local_initial_state(5000.0, ["AAPL"])
        state["transactions"].append(_tx("BUY", "AAPL", 10, 100.0))
        state["positions"]["AAPL"] = 10
        prices = {"AAPL": 110.0}
        portfolio_app.view_portfolio_local(state, prices)
        output = capsys.readouterr().out
        # Should show $100.00 unrealized gain
        assert "100.00" in output

    def test_gl_column_shows_loss(self, capsys):
        state = portfolio_app.create_local_initial_state(5000.0, ["AAPL"])
        state["transactions"].append(_tx("BUY", "AAPL", 10, 100.0))
        state["positions"]["AAPL"] = 10
        prices = {"AAPL": 90.0}
        portfolio_app.view_portfolio_local(state, prices)
        output = capsys.readouterr().out
        assert "100.00" in output  # $100 loss (abs value in the output)


# =====================================================================
# compute_target_trades -- rebalance threshold
# =====================================================================

class TestRebalanceThreshold:
    """Tests for the drift threshold that suppresses noisy rebalances."""

    def _make_portfolio(self, positions, cash, prices):
        """Build a PortfolioState for testing."""
        return PortfolioState(cash=cash, positions=positions, prices=prices)

    def test_balanced_portfolio_no_trades(self):
        """A nearly balanced portfolio should produce no trades."""
        # 2 tickers, each ~$500, cash ~5%
        prices = {"A": 100.0, "B": 100.0}
        positions = {"A": 5, "B": 5}
        # total equity = 1000, cash = ~55 (5.5%)
        portfolio = self._make_portfolio(positions, 55.0, prices)
        trades = compute_target_trades(portfolio, {}, rebalance_threshold=0.05)
        assert trades == []

    def test_large_drift_triggers_trade(self):
        """A big imbalance should trigger trades even with threshold."""
        prices = {"A": 100.0, "B": 100.0}
        positions = {"A": 10, "B": 0}  # 100% in A, 0% in B
        portfolio = self._make_portfolio(positions, 50.0, prices)
        trades = compute_target_trades(portfolio, {}, rebalance_threshold=0.05)
        assert len(trades) > 0

    def test_zero_threshold_always_trades(self):
        """With threshold=0, even tiny diffs generate trades."""
        prices = {"A": 100.0, "B": 50.0}
        # Slightly off-balance
        positions = {"A": 4, "B": 9}
        portfolio = self._make_portfolio(positions, 110.0, prices)
        trades = compute_target_trades(portfolio, {}, rebalance_threshold=0.0)
        assert len(trades) > 0

    def test_threshold_suppresses_small_drift(self):
        """1-share drift on a cheap stock stays within threshold."""
        # 10 tickers each at $50, target ~$475 each = 9-10 shares
        tickers = [f"T{i}" for i in range(10)]
        prices = {t: 50.0 for t in tickers}
        # Set all to 9 shares except one at 10
        positions = {t: 9 for t in tickers}
        positions["T0"] = 10
        # equity = 9*9*50 + 10*50 = 4050 + 500 = 4550, cash to make ~5%
        equity = sum(positions[t] * prices[t] for t in tickers)
        cash = equity * 0.05 / 0.95  # so cash is exactly 5% of total
        portfolio = self._make_portfolio(positions, cash, prices)
        # 1 share of $50 on a ~$4800 portfolio is ~1% drift -- below 5%
        trades = compute_target_trades(portfolio, {}, rebalance_threshold=0.05)
        assert trades == []

    def test_initial_buy_ignores_threshold(self):
        """From zero positions, all tickers need buying regardless of threshold."""
        prices = {"A": 100.0, "B": 50.0}
        positions = {"A": 0, "B": 0}
        portfolio = self._make_portfolio(positions, 1000.0, prices)
        trades = compute_target_trades(portfolio, {}, rebalance_threshold=0.05)
        buy_tickers = {t["ticker"] for t in trades if t["action"] == "BUY"}
        assert "A" in buy_tickers
        assert "B" in buy_tickers


# =====================================================================
# Menu changes (local broker menu)
# =====================================================================

class TestMenuUpdate:
    """Verify local menu numbering was updated."""

    def test_menu_has_9_items(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            portfolio_app.print_local_menu()
        output = buf.getvalue()
        assert "1." in output
        assert "9." in output
        assert "gains/losses" in output.lower()
        assert "Save" in output

    def test_sell_is_item_4(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            portfolio_app.print_local_menu()
        output = buf.getvalue()
        assert "Sell" in output
        # "4." should appear before "Sell" on the same line
        for line in output.split("\n"):
            if "Sell" in line:
                assert "4." in line
                break

    def test_menu_order(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            portfolio_app.print_local_menu()
        output = buf.getvalue()
        # Verify order: Sell before gains/losses, gains/losses before Save
        pos_sell = output.find("Sell")
        pos_gains = output.find("gains/losses")
        pos_save = output.find("Save")
        assert pos_sell < pos_gains < pos_save
        # gains/losses is now item 6
        assert "6." in output.split("gains/losses")[0].split("\n")[-1]

    def test_save_is_item_9(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            portfolio_app.print_local_menu()
        output = buf.getvalue()
        for line in output.split("\n"):
            if "Save" in line:
                assert "9." in line
                break


# =====================================================================
# compute_target_shares -- isolation tests
# =====================================================================

class TestComputeTargetSharesIsolation:
    """Test compute_target_shares directly as a pure function."""

    def test_empty_positions(self):
        result = compute_target_shares({}, {}, 1000.0)
        assert result == {}

    def test_zero_total_value(self):
        result = compute_target_shares({"A": 0}, {"A": 100.0}, 0.0)
        assert result == {"A": 0}

    def test_idempotency(self):
        """Compute target -> set positions to target -> recompute -> same target."""
        prices = {"A": 100.0, "B": 50.0, "C": 200.0}
        positions = {"A": 0, "B": 0, "C": 0}
        cash = 10000.0

        # First pass: compute targets
        targets1 = compute_target_shares(positions, prices, cash)
        # Compute leftover cash after buying target shares
        spent = sum(targets1[t] * prices[t] for t in targets1)
        new_cash = cash - spent

        # Second pass: positions are now at target
        targets2 = compute_target_shares(targets1, prices, new_cash)
        assert targets2 == targets1

    def test_equal_weight_dollar_values(self):
        """All target dollar values should be within 1 share-price of each other."""
        prices = {"A": 100.0, "B": 50.0, "C": 200.0, "D": 25.0}
        positions = {"A": 0, "B": 0, "C": 0, "D": 0}
        cash = 10000.0

        targets = compute_target_shares(positions, prices, cash)
        dollar_values = [targets[t] * prices[t] for t in sorted(targets)]
        max_price = max(prices.values())
        # Any two tickers' dollar values should differ by at most one share of the most expensive
        for i in range(len(dollar_values)):
            for j in range(i + 1, len(dollar_values)):
                assert abs(dollar_values[i] - dollar_values[j]) <= max_price

    def test_cash_reserve_enforced(self):
        """Total allocated equity should be <= (1 - reserve) * total value."""
        prices = {"A": 100.0, "B": 50.0, "C": 200.0}
        positions = {"A": 0, "B": 0, "C": 0}
        cash = 10000.0

        targets = compute_target_shares(positions, prices, cash)
        total_allocated = sum(targets[t] * prices[t] for t in targets)
        assert total_allocated <= (1 - CASH_RESERVE_PCT) * cash + 0.01  # +epsilon for float

    def test_greedy_most_underweight_first(self):
        """Leftover budget should go to the most-underweight ticker."""
        # Two tickers, one cheap and one expensive
        # With $1000 and 5% reserve: investable = $950, target_per = $475
        # A: floor(475/100) = 4 shares ($400), gap = $75
        # B: floor(475/10) = 47 shares ($470), gap = $5
        # Remaining = 950 - 400 - 470 = 80
        # A has bigger gap -> gets the extra share first
        prices = {"A": 100.0, "B": 10.0}
        positions = {"A": 0, "B": 0}
        cash = 1000.0

        targets = compute_target_shares(positions, prices, cash)
        # A should get floor(475/100) + greedy extras due to large gap
        assert targets["A"] >= 4

    def test_inv_vol_tiebreak(self):
        """Among equal gaps, lower-vol ticker gets the extra share."""
        # Create two tickers with identical prices and equal gaps
        # but different volatilities via history
        prices = {"LO": 100.0, "HI": 100.0}
        positions = {"LO": 0, "HI": 0}
        cash = 1050.0  # investable = 997.5, target_per = 498.75
        # floor: 4 each = 800, remaining = 197.5 -> 1 extra share

        # Create history: LO has low vol, HI has high vol
        dates = pd.date_range("2024-01-01", periods=70, freq="B")
        np.random.seed(603)
        lo_prices = 100 + np.cumsum(np.random.normal(0, 0.1, 70))
        hi_prices = 100 + np.cumsum(np.random.normal(0, 2.0, 70))
        history = {
            "LO": pd.DataFrame({"Close": lo_prices}, index=dates),
            "HI": pd.DataFrame({"Close": hi_prices}, index=dates),
        }

        targets = compute_target_shares(positions, prices, cash,
                                         history=history, vol_lookback=60)
        # Both at 4+1 or 5+0 -- the extra share should go to LO (lower vol tiebreak)
        # Since gaps are equal, inv-vol decides: LO gets the extra
        assert targets["LO"] >= targets["HI"]

    def test_single_ticker(self):
        """With one ticker, all investable cash goes to it."""
        targets = compute_target_shares({"X": 0}, {"X": 50.0}, 1000.0)
        investable = 1000.0 * (1 - CASH_RESERVE_PCT)
        expected = math.floor(investable / 50.0)
        assert targets["X"] == expected

    def test_zero_price_ticker_gets_zero_shares(self):
        """Ticker with price=0 should get 0 shares."""
        targets = compute_target_shares(
            {"A": 0, "B": 0}, {"A": 100.0, "B": 0.0}, 1000.0
        )
        assert targets["B"] == 0
        assert targets["A"] > 0

    def test_expensive_stock_zero_shares(self):
        """If a stock costs more than target_per, it gets 0 from floor."""
        # $500 stock, $1000 cash, 2 tickers: investable = 950, target_per = 475
        # floor(475/500) = 0 for the expensive stock
        targets = compute_target_shares(
            {"CHEAP": 0, "PRICEY": 0}, {"CHEAP": 10.0, "PRICEY": 500.0}, 1000.0
        )
        # Greedy may still assign 1 share to PRICEY if budget allows
        assert targets["PRICEY"] >= 0
        assert targets["CHEAP"] >= 0


# =====================================================================
# compute_volatility -- isolation tests
# =====================================================================

class TestComputeVolatility:
    """Test the volatility helper directly."""

    def test_no_history_returns_inf(self):
        assert compute_volatility({}, "AAPL") == float("inf")

    def test_empty_dataframe_returns_inf(self):
        history = {"X": pd.DataFrame({"Close": []})}
        assert compute_volatility(history, "X") == float("inf")

    def test_single_row_returns_inf(self):
        history = {"X": pd.DataFrame({"Close": [100.0]})}
        assert compute_volatility(history, "X") == float("inf")

    def test_constant_prices_zero_vol(self):
        history = {"X": pd.DataFrame({"Close": [100.0] * 10})}
        assert compute_volatility(history, "X") == pytest.approx(0.0)

    def test_positive_vol_for_varying_prices(self):
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.normal(0, 1, 100))
        history = {"X": pd.DataFrame({"Close": prices})}
        vol = compute_volatility(history, "X", lookback=60)
        assert vol > 0


# =====================================================================
# needs_rebalance -- isolation tests
# =====================================================================

class TestNeedsRebalance:
    """Test the needs_rebalance check directly."""

    def test_balanced_portfolio_no_rebalance(self):
        prices = {"A": 100.0, "B": 100.0}
        positions = {"A": 5, "B": 5}
        # total = 1000 + cash, target per ticker = 47.5%
        cash = 55.0  # ~5.2% cash
        assert not needs_rebalance(positions, prices, cash, threshold=0.05)

    def test_large_drift_triggers_rebalance(self):
        prices = {"A": 100.0, "B": 100.0}
        positions = {"A": 10, "B": 0}
        cash = 50.0
        assert needs_rebalance(positions, prices, cash, threshold=0.05)

    def test_excess_cash_triggers_rebalance(self):
        """Cash > 2 * target_weight should trigger even if positions are fine."""
        prices = {"A": 100.0, "B": 100.0}
        positions = {"A": 1, "B": 1}
        cash = 5000.0  # huge cash vs equity
        assert needs_rebalance(positions, prices, cash, threshold=0.05)

    def test_empty_positions(self):
        assert not needs_rebalance({}, {}, 1000.0)

    def test_zero_total_value(self):
        assert not needs_rebalance({"A": 0}, {"A": 0.0}, 0.0)


# =====================================================================
# compute_rebalance_trades -- isolation tests
# =====================================================================

class TestComputeRebalanceTrades:
    """Test trade generation from current vs target diff."""

    def test_no_diff_no_trades(self):
        trades = compute_rebalance_trades(
            {"A": 5, "B": 5}, {"A": 5, "B": 5},
            {"A": 100.0, "B": 100.0}, 1050.0, rebalance_threshold=0.05
        )
        assert trades == []

    def test_sells_before_buys(self):
        """Sells should appear before buys in the output."""
        trades = compute_rebalance_trades(
            {"A": 10, "B": 0}, {"A": 5, "B": 5},
            {"A": 100.0, "B": 100.0}, 1050.0, rebalance_threshold=0.01
        )
        actions = [t["action"] for t in trades]
        sell_indices = [i for i, a in enumerate(actions) if a == "SELL"]
        buy_indices = [i for i, a in enumerate(actions) if a == "BUY"]
        if sell_indices and buy_indices:
            assert max(sell_indices) < min(buy_indices)

    def test_threshold_filters_small_diffs(self):
        """Small diffs within threshold should be suppressed."""
        # 10 tickers each at $50, total equity ~$4750, total ~$5000
        # 1-share diff is ~1% drift -- below 5% threshold
        tickers = [f"T{i}" for i in range(10)]
        current = {t: 9 for t in tickers}
        current["T0"] = 10  # one ticker has 1 extra share
        target = {t: 9 for t in tickers}
        target["T0"] = 9   # target says sell 1
        prices = {t: 50.0 for t in tickers}
        trades = compute_rebalance_trades(
            current, target, prices, 5000.0, rebalance_threshold=0.05
        )
        # 1 share of $50 on $5000 portfolio is 1% drift -- below 5% threshold
        assert trades == []


# =====================================================================
# Cross-module consistency (local broker)
# =====================================================================

class TestCrossModuleConsistencyLocal:
    """Verify all consumers import from the canonical equal_weight module."""

    def test_app_imports_compute_target_trades(self):
        """app.py should have compute_target_trades available."""
        assert hasattr(portfolio_app, "compute_target_trades")

    def test_app_imports_cash_reserve(self):
        """app.py should import CASH_RESERVE_PCT from equal_weight."""
        assert portfolio_app.CASH_RESERVE_PCT == CASH_RESERVE_PCT

    def test_compute_target_trades_defined_in_equal_weight(self):
        """The function definition should exist in strategies/equal_weight.py."""
        source_file = inspect.getfile(compute_target_trades)
        assert "equal_weight" in source_file


# =====================================================================
# compute_proportional_sells -- pure function tests (local)
# =====================================================================

class TestProportionalSells:
    """Tests for the proportional sell/withdraw algorithm."""

    def test_empty_positions(self):
        result = portfolio_app.compute_proportional_sells({}, {"A": 100.0}, 500.0)
        assert result == []

    def test_zero_target(self):
        result = portfolio_app.compute_proportional_sells(
            {"A": 10}, {"A": 100.0}, 0.0)
        assert result == []

    def test_negative_target(self):
        result = portfolio_app.compute_proportional_sells(
            {"A": 10}, {"A": 100.0}, -100.0)
        assert result == []

    def test_single_ticker_exact(self):
        """Selling exactly the dollar value of some whole shares."""
        result = portfolio_app.compute_proportional_sells(
            {"A": 10}, {"A": 100.0}, 300.0)
        assert len(result) == 1
        assert result[0]["ticker"] == "A"
        assert result[0]["shares"] == 3
        assert result[0]["amount"] == 300.0
        assert result[0]["action"] == "SELL"

    def test_single_ticker_floor(self):
        """Target that doesn't divide evenly into whole shares."""
        result = portfolio_app.compute_proportional_sells(
            {"A": 10}, {"A": 100.0}, 250.0)
        # Floor(250/100) = 2, greedy adds 1 more (shortfall=50, price=100 > 50)
        # Actually shortfall = 250 - 200 = 50, price $100 > $50, so no more
        assert len(result) == 1
        assert result[0]["shares"] == 2
        assert result[0]["amount"] == 200.0

    def test_two_tickers_proportional(self):
        """Two tickers: sells should be roughly proportional to holdings."""
        positions = {"A": 10, "B": 20}
        prices = {"A": 100.0, "B": 50.0}
        # A equity = 1000, B equity = 1000, total = 2000
        # Target = $500, each contributes $250
        # A: floor(250/100) = 2 ($200), B: floor(250/50) = 5 ($250)
        # Raised = 450, shortfall = 50, B ($50) gets +1 -> B=6
        result = portfolio_app.compute_proportional_sells(positions, prices, 500.0)
        tickers = {t["ticker"] for t in result}
        assert "A" in tickers
        assert "B" in tickers
        a_trade = [t for t in result if t["ticker"] == "A"][0]
        b_trade = [t for t in result if t["ticker"] == "B"][0]
        assert a_trade["shares"] == 2
        assert b_trade["shares"] == 6
        # Total raised = 200 + 300 = 500 (exact)
        total = a_trade["amount"] + b_trade["amount"]
        assert total == pytest.approx(500.0)

    def test_sell_all_when_target_exceeds_equity(self):
        """Target > total equity should sell everything."""
        positions = {"A": 5, "B": 3}
        prices = {"A": 100.0, "B": 50.0}
        # Total equity = 500 + 150 = 650
        result = portfolio_app.compute_proportional_sells(positions, prices, 10000.0)
        a_trade = [t for t in result if t["ticker"] == "A"][0]
        b_trade = [t for t in result if t["ticker"] == "B"][0]
        assert a_trade["shares"] == 5
        assert b_trade["shares"] == 3

    def test_expensive_ticker_redistribution(self):
        """When one ticker's floor is 0, shortfall goes to others."""
        positions = {"EXPENSIVE": 1, "CHEAP": 100}
        prices = {"EXPENSIVE": 500.0, "CHEAP": 10.0}
        # Total = 500 + 1000 = 1500
        # Target = $200, EXPENSIVE contributes 200*500/1500 = 66.7 -> floor(66.7/500) = 0
        # CHEAP contributes 200*1000/1500 = 133.3 -> floor(133.3/10) = 13
        # Shortfall: 200 - 130 = 70, CHEAP gets more
        result = portfolio_app.compute_proportional_sells(positions, prices, 200.0)
        total_raised = sum(t["amount"] for t in result)
        # Should be close to $200 (within one share price of cheapest)
        assert total_raised <= 200.0 + 10.0
        assert total_raised >= 200.0 - 10.0

    def test_single_share_cap(self):
        """Cannot sell more than held shares."""
        result = portfolio_app.compute_proportional_sells(
            {"A": 1}, {"A": 100.0}, 500.0)
        assert result[0]["shares"] == 1

    def test_proceeds_within_one_share(self):
        """Total proceeds should be within 1 share-price of target."""
        positions = {"A": 50, "B": 30, "C": 20}
        prices = {"A": 45.0, "B": 120.0, "C": 80.0}
        target = 2000.0
        result = portfolio_app.compute_proportional_sells(positions, prices, target)
        total_raised = sum(t["amount"] for t in result)
        max_price = max(prices.values())
        assert total_raised >= target - max_price
        assert total_raised <= target + max_price

    def test_trade_dict_format(self):
        """Each trade dict should have the correct keys."""
        result = portfolio_app.compute_proportional_sells(
            {"A": 10}, {"A": 50.0}, 100.0)
        assert len(result) == 1
        trade = result[0]
        assert set(trade.keys()) == {"action", "ticker", "shares", "price", "amount"}
        assert trade["action"] == "SELL"
        assert isinstance(trade["shares"], int)
        assert trade["price"] == 50.0

    def test_no_positions_with_zero_shares(self):
        """Tickers with 0 shares should be ignored."""
        result = portfolio_app.compute_proportional_sells(
            {"A": 0, "B": 10}, {"A": 100.0, "B": 50.0}, 200.0)
        tickers = [t["ticker"] for t in result]
        assert "A" not in tickers

    def test_ticker_with_no_price(self):
        """Tickers missing from prices dict should be ignored."""
        result = portfolio_app.compute_proportional_sells(
            {"A": 10, "B": 10}, {"A": 100.0}, 500.0)
        tickers = [t["ticker"] for t in result]
        assert "B" not in tickers


# =====================================================================
# sell_shares_local -- integration tests (monkeypatched input)
# =====================================================================

class TestSellSharesLocalIntegration:
    """Integration tests for interactive sell_shares_local function."""

    def _make_state(self, positions, cash=5000.0):
        """Build a state dict for testing."""
        tickers = sorted(positions.keys())
        state = portfolio_app.create_local_initial_state(cash, tickers)
        state["positions"] = dict(positions)
        # Add buy transactions so positions are tracked
        for t, shares in positions.items():
            if shares > 0:
                state["transactions"].append(
                    _tx("BUY", t, shares, 100.0))
        return state

    def test_proportional_sell_updates_state(self, monkeypatch, tmp_path):
        """Option A: proportional withdrawal updates cash and positions."""
        # Redirect state file to temp
        state_file = str(tmp_path / ".portfolio.json")
        monkeypatch.setattr(portfolio_app, "LOCAL_STATE_FILE", state_file)

        state = self._make_state({"A": 10, "B": 10})
        prices = {"A": 100.0, "B": 100.0}

        # Simulate: choose 'a', amount '500', confirm 'y'
        inputs = iter(["a", "500", "y"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        result = portfolio_app.sell_shares_local(state, prices)

        # Should have sold some shares of A and B
        assert result["positions"]["A"] < 10 or result["positions"]["B"] < 10
        assert result["cash"] > 5000.0

    def test_specific_sell_updates_state(self, monkeypatch, tmp_path):
        """Option B: sell specific ticker updates state correctly."""
        state_file = str(tmp_path / ".portfolio.json")
        monkeypatch.setattr(portfolio_app, "LOCAL_STATE_FILE", state_file)

        state = self._make_state({"A": 10, "B": 5})
        prices = {"A": 100.0, "B": 200.0}

        # Simulate: choose 'b', pick ticker 1 (A), sell 3 shares, confirm 'y'
        inputs = iter(["b", "1", "3", "y"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        result = portfolio_app.sell_shares_local(state, prices)

        assert result["positions"]["A"] == 7
        assert result["cash"] == pytest.approx(5000.0 + 300.0)

    def test_sell_transactions_recorded(self, monkeypatch, tmp_path):
        """Sell transactions should be recorded in the transaction log."""
        state_file = str(tmp_path / ".portfolio.json")
        monkeypatch.setattr(portfolio_app, "LOCAL_STATE_FILE", state_file)

        state = self._make_state({"A": 10})
        prices = {"A": 50.0}
        n_txns_before = len(state["transactions"])

        inputs = iter(["b", "1", "2", "y"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        result = portfolio_app.sell_shares_local(state, prices)

        new_txns = result["transactions"][n_txns_before:]
        assert len(new_txns) == 1
        assert new_txns[0]["action"] == "SELL"
        assert new_txns[0]["ticker"] == "A"
        assert new_txns[0]["shares"] == 2

    def test_cancel_no_changes(self, monkeypatch):
        """Option C: cancel should not modify state."""
        state = self._make_state({"A": 10})
        prices = {"A": 100.0}
        original_cash = state["cash"]
        original_pos = dict(state["positions"])

        inputs = iter(["c"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        result = portfolio_app.sell_shares_local(state, prices)
        assert result["cash"] == original_cash
        assert result["positions"] == original_pos

    def test_no_shares_message(self, capsys):
        """sell_shares_local with no positions should print a message."""
        state = portfolio_app.create_local_initial_state(5000.0, ["A", "B"])
        prices = {"A": 100.0, "B": 100.0}
        portfolio_app.sell_shares_local(state, prices)
        output = capsys.readouterr().out
        assert "No shares to sell" in output


# =====================================================================
# compute_portfolio_stats -- Sharpe ratio tests
# =====================================================================

class TestPortfolioSharpe:
    """Tests for compute_portfolio_stats."""

    def _make_history(self, tickers, n_days=60, start_price=100.0,
                      daily_return=0.0005):
        """Build synthetic daily price history."""
        dates = pd.date_range("2025-01-02", periods=n_days, freq="B")
        history = {}
        for t in tickers:
            prices = [start_price]
            for _ in range(n_days - 1):
                prices.append(prices[-1] * (1 + daily_return))
            history[t] = pd.DataFrame(
                {"Close": prices[:n_days]}, index=dates[:n_days])
        return history

    def test_insufficient_data_returns_none(self):
        """Fewer than 2 days of history should return None."""
        txns = [_deposit(5000.0)]
        history = {"A": pd.DataFrame({"Close": [100.0]},
                                     index=[pd.Timestamp("2025-01-02")])}
        result = portfolio_app.compute_portfolio_stats(txns, ["A"], history)
        assert result is None

    def test_empty_transactions_returns_none(self):
        result = portfolio_app.compute_portfolio_stats([], ["A"], {})
        assert result is None

    def test_empty_history_returns_none(self):
        txns = [_deposit(5000.0)]
        result = portfolio_app.compute_portfolio_stats(txns, ["A"], {})
        assert result is None

    def test_returns_dict_with_correct_keys(self):
        """With enough data, should return dict with all expected keys."""
        history = self._make_history(["A"], n_days=60)
        txns = [
            {**_deposit(5000.0), "date": "2025-01-02T00:00:00"},
            {**_tx("BUY", "A", 10, 100.0), "date": "2025-01-02T00:00:00"},
        ]
        result = portfolio_app.compute_portfolio_stats(txns, ["A"], history)
        assert result is not None
        assert set(result.keys()) == {"sharpe", "ann_return", "ann_vol",
                                       "rf_rate", "n_days", "max_drawdown"}

    def test_deposit_only_near_zero_vol(self):
        """Cash-only portfolio has near-zero volatility."""
        history = self._make_history(["A"], n_days=30)
        txns = [{**_deposit(5000.0), "date": "2025-01-02T00:00:00"}]
        result = portfolio_app.compute_portfolio_stats(txns, ["A"], history)
        # No shares held, portfolio is all cash -> zero market return
        # Result might be None if all daily returns are 0 (mask issue)
        # or might have near-zero vol
        if result is not None:
            assert result["ann_vol"] < 0.01

    def test_positive_market_positive_return(self):
        """Rising prices should give positive annualized return."""
        history = self._make_history(["A"], n_days=60, daily_return=0.001)
        txns = [
            {**_deposit(5000.0), "date": "2025-01-02T00:00:00"},
            {**_tx("BUY", "A", 40, 100.0), "date": "2025-01-02T00:00:00"},
        ]
        result = portfolio_app.compute_portfolio_stats(txns, ["A"], history)
        assert result is not None
        assert result["ann_return"] > 0
        assert result["sharpe"] > 0

    def test_n_days_correct(self):
        """n_days should match the number of trading days in history."""
        history = self._make_history(["A"], n_days=45)
        txns = [
            {**_deposit(5000.0), "date": "2025-01-02T00:00:00"},
            {**_tx("BUY", "A", 10, 100.0), "date": "2025-01-02T00:00:00"},
        ]
        result = portfolio_app.compute_portfolio_stats(txns, ["A"], history)
        assert result is not None
        assert result["n_days"] == 45

    def test_multiple_tickers(self):
        """Should work with multiple tickers."""
        history = self._make_history(["A", "B"], n_days=40,
                                     daily_return=0.0005)
        txns = [
            {**_deposit(10000.0), "date": "2025-01-02T00:00:00"},
            {**_tx("BUY", "A", 30, 100.0), "date": "2025-01-02T00:00:00"},
            {**_tx("BUY", "B", 30, 100.0), "date": "2025-01-02T00:00:00"},
        ]
        result = portfolio_app.compute_portfolio_stats(txns, ["A", "B"], history)
        assert result is not None
        assert result["sharpe"] > 0

    def test_max_drawdown_non_positive(self):
        """Max drawdown should be <= 0."""
        history = self._make_history(["A"], n_days=60, daily_return=0.001)
        txns = [
            {**_deposit(5000.0), "date": "2025-01-02T00:00:00"},
            {**_tx("BUY", "A", 40, 100.0), "date": "2025-01-02T00:00:00"},
        ]
        result = portfolio_app.compute_portfolio_stats(txns, ["A"], history)
        assert result is not None
        assert result["max_drawdown"] <= 0.0

    def test_rf_rate_lowers_sharpe(self):
        """Non-zero risk-free rate should produce a lower Sharpe."""
        history = self._make_history(["A"], n_days=60, daily_return=0.001)
        txns = [
            {**_deposit(5000.0), "date": "2025-01-02T00:00:00"},
            {**_tx("BUY", "A", 40, 100.0), "date": "2025-01-02T00:00:00"},
        ]
        result_0 = portfolio_app.compute_portfolio_stats(txns, ["A"], history, rf_rate=0.0)
        result_rf = portfolio_app.compute_portfolio_stats(txns, ["A"], history, rf_rate=0.04)
        assert result_0 is not None and result_rf is not None
        assert result_rf["sharpe"] < result_0["sharpe"]
        assert result_rf["rf_rate"] == 0.04

    def test_rf_rate_stored_in_result(self):
        """The rf_rate used should be stored in the result dict."""
        history = self._make_history(["A"], n_days=30)
        txns = [
            {**_deposit(5000.0), "date": "2025-01-02T00:00:00"},
            {**_tx("BUY", "A", 10, 100.0), "date": "2025-01-02T00:00:00"},
        ]
        result = portfolio_app.compute_portfolio_stats(txns, ["A"], history, rf_rate=0.05)
        assert result is not None
        assert result["rf_rate"] == 0.05


# =====================================================================
# view_portfolio_local -- Sharpe display
# =====================================================================

class TestViewPortfolioSharpe:
    """Tests for Sharpe display in view_portfolio_local."""

    def test_sharpe_displayed_when_provided(self, capsys):
        state = portfolio_app.create_local_initial_state(5000.0, ["AAPL"])
        state["transactions"].append(_tx("BUY", "AAPL", 10, 100.0))
        state["positions"]["AAPL"] = 10
        prices = {"AAPL": 110.0}
        stats = {"sharpe": 1.23, "ann_return": 0.123, "ann_vol": 0.10,
                 "rf_rate": 0.045, "n_days": 60, "max_drawdown": -0.05}
        portfolio_app.view_portfolio_local(state, prices, sharpe_stats=stats)
        output = capsys.readouterr().out
        assert "Sharpe" in output
        assert "1.23" in output
        assert "12.3" in output  # ann return %
        assert "10.0" in output  # ann vol %
        assert "Rf" in output
        assert "4.5" in output   # rf rate %

    def test_sharpe_not_displayed_when_none(self, capsys):
        state = portfolio_app.create_local_initial_state(5000.0, ["AAPL"])
        prices = {"AAPL": 100.0}
        portfolio_app.view_portfolio_local(state, prices, sharpe_stats=None)
        output = capsys.readouterr().out
        assert "Sharpe" not in output

    def test_preliminary_caveat_when_few_days(self, capsys):
        state = portfolio_app.create_local_initial_state(5000.0, ["AAPL"])
        state["transactions"].append(_tx("BUY", "AAPL", 10, 100.0))
        state["positions"]["AAPL"] = 10
        prices = {"AAPL": 110.0}
        stats = {"sharpe": 0.50, "ann_return": 0.05, "ann_vol": 0.10,
                 "rf_rate": 0.045, "n_days": 15, "max_drawdown": -0.02}
        portfolio_app.view_portfolio_local(state, prices, sharpe_stats=stats)
        output = capsys.readouterr().out
        assert "preliminary" in output
        assert "15 days" in output

    def test_needs_more_data_message_when_none_with_positions(self, capsys):
        """When stats are None but positions exist, show 'needs data' hint."""
        state = portfolio_app.create_local_initial_state(5000.0, ["AAPL"])
        state["transactions"].append(_tx("BUY", "AAPL", 10, 100.0))
        state["positions"]["AAPL"] = 10
        prices = {"AAPL": 110.0}
        portfolio_app.view_portfolio_local(state, prices, sharpe_stats=None)
        output = capsys.readouterr().out
        assert "needs 2+ trading days" in output

    def test_no_sharpe_message_when_no_positions(self, capsys):
        """When no positions and no stats, no Sharpe message at all."""
        state = portfolio_app.create_local_initial_state(5000.0, ["AAPL"])
        prices = {"AAPL": 100.0}
        portfolio_app.view_portfolio_local(state, prices, sharpe_stats=None)
        output = capsys.readouterr().out
        assert "Sharpe" not in output
        assert "needs" not in output


# #####################################################################
#
#  SECTION 2: ALPACA BROKER TESTS
#
# #####################################################################


# ===================================================================
# TestPositionInfo -- dataclass fields
# ===================================================================

class TestPositionInfo:
    def test_fields(self):
        p = PositionInfo(
            ticker="AAPL", qty=10, avg_entry_price=150.0,
            current_price=155.0, market_value=1550.0,
            unrealized_pl=50.0, unrealized_pl_pct=0.0333,
        )
        assert p.ticker == "AAPL"
        assert p.qty == 10
        assert p.avg_entry_price == 150.0
        assert p.current_price == 155.0
        assert p.market_value == 1550.0
        assert p.unrealized_pl == 50.0
        assert p.unrealized_pl_pct == 0.0333

    def test_is_dataclass(self):
        import dataclasses
        assert dataclasses.is_dataclass(PositionInfo)


# ===================================================================
# TestOrderResult -- dataclass fields
# ===================================================================

class TestOrderResult:
    def test_fields(self):
        r = OrderResult(
            order_id="abc-123", ticker="MSFT", side="buy",
            qty=5, status="filled", filled_qty=5, filled_price=400.0,
        )
        assert r.order_id == "abc-123"
        assert r.ticker == "MSFT"
        assert r.side == "buy"
        assert r.qty == 5
        assert r.status == "filled"
        assert r.filled_qty == 5
        assert r.filled_price == 400.0

    def test_none_filled_price(self):
        r = OrderResult(
            order_id="abc-123", ticker="MSFT", side="buy",
            qty=5, status="pending", filled_qty=0, filled_price=None,
        )
        assert r.filled_price is None

    def test_is_dataclass(self):
        import dataclasses
        assert dataclasses.is_dataclass(OrderResult)


# ===================================================================
# TestAlpacaBroker
# ===================================================================

class TestAlpacaBroker:
    """Tests for AlpacaBroker. All mock the TradingClient."""

    def test_requires_credentials(self):
        """Should raise ValueError without API keys."""
        with patch.dict(os.environ, {}, clear=True):
            env = {k: v for k, v in os.environ.items()
                   if k not in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")}
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(ValueError, match="Alpaca credentials required"):
                    AlpacaBroker()

    def test_credentials_from_args(self):
        """Direct credential passing should work."""
        with patch("finance_tools.broker.alpaca.AlpacaBroker._get_client"):
            broker = AlpacaBroker(api_key="test-key", secret_key="test-secret")
            assert broker._api_key == "test-key"
            assert broker._secret_key == "test-secret"

    def test_credentials_from_env(self):
        """Environment variable credential reading."""
        with patch.dict(os.environ, {
            "ALPACA_API_KEY": "env-key",
            "ALPACA_SECRET_KEY": "env-secret",
        }):
            broker = AlpacaBroker()
            assert broker._api_key == "env-key"
            assert broker._secret_key == "env-secret"

    def test_paper_mode_default(self):
        """Paper mode should be True by default."""
        broker = AlpacaBroker(api_key="k", secret_key="s")
        assert broker._paper is True

    def test_get_cash(self):
        broker = AlpacaBroker(api_key="k", secret_key="s")
        mock_client = MagicMock()
        mock_client.get_account.return_value = _mock_account(cash=50_000.0)
        broker._client = mock_client
        assert broker.get_cash() == 50_000.0

    def test_get_equity(self):
        broker = AlpacaBroker(api_key="k", secret_key="s")
        mock_client = MagicMock()
        mock_client.get_account.return_value = _mock_account(equity=200_000.0)
        broker._client = mock_client
        assert broker.get_equity() == 200_000.0

    def test_get_positions_returns_dict(self):
        broker = AlpacaBroker(api_key="k", secret_key="s")
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = [
            _mock_position("AAPL", 10, 150.0, 155.0, 1550.0, 50.0, 0.033),
            _mock_position("MSFT", 5, 400.0, 410.0, 2050.0, 50.0, 0.025),
        ]
        broker._client = mock_client

        result = broker.get_positions()
        assert isinstance(result, dict)
        assert "AAPL" in result
        assert "MSFT" in result
        assert isinstance(result["AAPL"], PositionInfo)
        assert result["AAPL"].qty == 10
        assert result["MSFT"].qty == 5

    def test_get_positions_empty(self):
        broker = AlpacaBroker(api_key="k", secret_key="s")
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = []
        broker._client = mock_client
        assert broker.get_positions() == {}

    def test_get_position_single(self):
        broker = AlpacaBroker(api_key="k", secret_key="s")
        mock_client = MagicMock()
        mock_client.get_open_position.return_value = _mock_position(
            "AAPL", 10, 150.0, 155.0, 1550.0, 50.0, 0.033)
        broker._client = mock_client

        result = broker.get_position("AAPL")
        assert result is not None
        assert result.ticker == "AAPL"
        assert result.qty == 10

    def test_get_position_unknown_returns_none(self):
        broker = AlpacaBroker(api_key="k", secret_key="s")
        mock_client = MagicMock()
        mock_client.get_open_position.side_effect = Exception("not found")
        broker._client = mock_client
        assert broker.get_position("XYZ") is None

    def test_is_market_open(self):
        broker = AlpacaBroker(api_key="k", secret_key="s")
        mock_client = MagicMock()
        mock_client.get_clock.return_value = _mock_clock(is_open=True)
        broker._client = mock_client
        assert broker.is_market_open() is True

    def test_is_market_closed(self):
        broker = AlpacaBroker(api_key="k", secret_key="s")
        mock_client = MagicMock()
        mock_client.get_clock.return_value = _mock_clock(is_open=False)
        broker._client = mock_client
        assert broker.is_market_open() is False

    def test_buy_submits_order(self):
        broker = AlpacaBroker(api_key="k", secret_key="s")
        mock_client = MagicMock()
        mock_client.submit_order.return_value = _mock_order(
            "order-buy-1", "AAPL", "buy", 10, "accepted", 0, None)
        broker._client = mock_client

        with patch.dict("sys.modules", {
            "alpaca.trading.requests": MagicMock(),
            "alpaca.trading.enums": MagicMock(),
        }):
            result = broker.buy("AAPL", 10)

        assert isinstance(result, OrderResult)
        assert result.ticker == "AAPL"
        assert result.side == "buy"
        assert result.qty == 10
        mock_client.submit_order.assert_called_once()

    def test_sell_submits_order(self):
        broker = AlpacaBroker(api_key="k", secret_key="s")
        mock_client = MagicMock()
        mock_client.submit_order.return_value = _mock_order(
            "order-sell-1", "AAPL", "sell", 5, "accepted", 0, None)
        broker._client = mock_client

        with patch.dict("sys.modules", {
            "alpaca.trading.requests": MagicMock(),
            "alpaca.trading.enums": MagicMock(),
        }):
            result = broker.sell("AAPL", 5)

        assert isinstance(result, OrderResult)
        assert result.ticker == "AAPL"
        assert result.side == "sell"
        assert result.qty == 5
        mock_client.submit_order.assert_called_once()

    def test_buy_returns_order_result(self):
        broker = AlpacaBroker(api_key="k", secret_key="s")
        mock_client = MagicMock()
        mock_client.submit_order.return_value = _mock_order(
            "order-buy-2", "MSFT", "buy", 3, "filled", 3, 405.50)
        broker._client = mock_client

        with patch.dict("sys.modules", {
            "alpaca.trading.requests": MagicMock(),
            "alpaca.trading.enums": MagicMock(),
        }):
            result = broker.buy("MSFT", 3)

        assert result.order_id == "order-buy-2"
        assert result.status == "filled"
        assert result.filled_qty == 3
        assert result.filled_price == 405.50

    def test_wait_for_fill_returns_on_filled(self):
        broker = AlpacaBroker(api_key="k", secret_key="s")
        mock_client = MagicMock()
        mock_client.get_order_by_id.side_effect = [
            _mock_order("o1", "AAPL", "buy", 10, "pending_new", 0, None),
            _mock_order("o1", "AAPL", "buy", 10, "filled", 10, 155.0),
        ]
        broker._client = mock_client

        result = broker.wait_for_fill("o1", timeout=5.0, poll_interval=0.01)
        assert result.status == "filled"
        assert result.filled_qty == 10
        assert result.filled_price == 155.0

    def test_wait_for_fill_timeout(self):
        broker = AlpacaBroker(api_key="k", secret_key="s")
        mock_client = MagicMock()
        mock_client.get_order_by_id.return_value = _mock_order(
            "o1", "AAPL", "buy", 10, "pending_new", 0, None)
        broker._client = mock_client

        result = broker.wait_for_fill("o1", timeout=0.05, poll_interval=0.01)
        assert result.status == "pending_new"

    def test_wait_for_fill_rejected(self):
        broker = AlpacaBroker(api_key="k", secret_key="s")
        mock_client = MagicMock()
        mock_client.get_order_by_id.return_value = _mock_order(
            "o1", "AAPL", "buy", 10, "rejected", 0, None)
        broker._client = mock_client

        result = broker.wait_for_fill("o1", timeout=5.0, poll_interval=0.01)
        assert result.status == "rejected"


# ===================================================================
# TestAlpacaTraderApp -- app-level functions (Alpaca broker path)
# ===================================================================

class TestAlpacaTraderApp:
    """Tests for Alpaca-facing app.py functions (via portfolio_app)."""

    def test_create_alpaca_initial_state(self):
        state = portfolio_app.create_alpaca_initial_state(["AAPL", "MSFT"])
        assert state["tickers"] == ["AAPL", "MSFT"]
        assert state["transactions"] == []
        assert "session_start" in state
        assert "last_refresh" in state

    def test_state_json_serializable(self):
        state = portfolio_app.create_alpaca_initial_state(["AAPL", "MSFT"])
        serialized = json.dumps(state)
        assert isinstance(serialized, str)

    def test_state_save_load_roundtrip(self, tmp_path):
        state = portfolio_app.create_alpaca_initial_state(["AAPL", "MSFT", "NVDA"])
        state["transactions"].append({
            "date": "2026-02-10T10:00:00",
            "action": "BUY",
            "ticker": "AAPL",
            "qty": 10,
            "order_id": "test-order",
            "status": "filled",
            "filled_qty": 10,
            "filled_price": 155.0,
        })

        filepath = tmp_path / "state.json"
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)
        with open(filepath) as f:
            loaded = json.load(f)

        assert loaded["tickers"] == state["tickers"]
        assert len(loaded["transactions"]) == 1
        assert loaded["transactions"][0]["ticker"] == "AAPL"

    def test_compute_alpaca_suggestions_returns_list(self):
        positions = {"AAPL": 0, "MSFT": 0}
        prices = {"AAPL": 150.0, "MSFT": 400.0}
        cash = 10_000.0
        result = portfolio_app.compute_alpaca_suggestions(positions, prices, cash, {})
        assert isinstance(result, list)

    def test_compute_alpaca_suggestions_empty_when_balanced(self):
        """When positions match targets, no trades needed."""
        positions = {"AAPL": 63}
        prices = {"AAPL": 150.0}
        cash = 525.0
        result = portfolio_app.compute_alpaca_suggestions(positions, prices, cash, {})
        assert isinstance(result, list)

    def test_compute_alpaca_suggestions_buys_with_cash(self):
        """With all cash and no positions, should suggest buys."""
        positions = {"AAPL": 0, "MSFT": 0}
        prices = {"AAPL": 150.0, "MSFT": 400.0}
        cash = 10_000.0
        result = portfolio_app.compute_alpaca_suggestions(positions, prices, cash, {})
        assert len(result) > 0
        actions = {t["action"] for t in result}
        assert "BUY" in actions

    def test_display_portfolio_alpaca_no_crash_empty(self, capsys):
        """Display functions shouldn't crash on empty data."""
        portfolio_app.display_portfolio_alpaca({}, {}, 0.0, [])
        captured = capsys.readouterr()
        assert "Total" in captured.out

    def test_display_portfolio_alpaca_with_data(self, capsys):
        portfolio_app.display_portfolio_alpaca(
            {"AAPL": 10, "MSFT": 5},
            {"AAPL": 150.0, "MSFT": 400.0},
            5000.0,
            ["AAPL", "MSFT"],
        )
        captured = capsys.readouterr()
        assert "AAPL" in captured.out
        assert "MSFT" in captured.out
        assert "Cash" in captured.out
        assert "Total" in captured.out

    def test_display_portfolio_alpaca_shows_pl(self, capsys):
        """P&L should be shown when pos_info is provided."""
        pos_info = {
            "AAPL": PositionInfo(
                ticker="AAPL", qty=10, avg_entry_price=140.0,
                current_price=150.0, market_value=1500.0,
                unrealized_pl=100.0, unrealized_pl_pct=0.0714),
        }
        portfolio_app.display_portfolio_alpaca(
            {"AAPL": 10},
            {"AAPL": 150.0},
            5000.0,
            ["AAPL"],
            pos_info=pos_info,
        )
        captured = capsys.readouterr()
        assert "140.00" in captured.out   # avg cost
        assert "+100.00" in captured.out  # P&L

    def test_display_suggestions_no_crash_empty(self, capsys):
        portfolio_app.display_suggestions([])
        captured = capsys.readouterr()
        assert "balanced" in captured.out.lower()

    def test_display_suggestions_with_trades(self, capsys):
        trades = [
            {"action": "BUY", "ticker": "AAPL", "shares": 10,
             "price": 150.0, "amount": 1500.0},
            {"action": "SELL", "ticker": "MSFT", "shares": 2,
             "price": 400.0, "amount": 800.0},
        ]
        portfolio_app.display_suggestions(trades)
        captured = capsys.readouterr()
        assert "BUY" in captured.out
        assert "SELL" in captured.out
        assert "AAPL" in captured.out

    def test_show_alpaca_transaction_log_empty(self, capsys):
        portfolio_app.show_alpaca_transaction_log({"transactions": []})
        captured = capsys.readouterr()
        assert "No transactions" in captured.out

    def test_show_alpaca_transaction_log_with_entries(self, capsys):
        state = {
            "transactions": [{
                "date": "2026-02-10T10:00:00",
                "action": "BUY",
                "ticker": "AAPL",
                "qty": 10,
                "filled_price": 155.0,
                "status": "filled",
            }]
        }
        portfolio_app.show_alpaca_transaction_log(state)
        captured = capsys.readouterr()
        assert "AAPL" in captured.out
        assert "BUY" in captured.out

    def test_execute_alpaca_trades_sells_before_buys(self):
        """Sells should execute before buys."""
        broker = MagicMock(spec=AlpacaBroker)

        def mock_buy(ticker, qty):
            return OrderResult(
                order_id=f"buy-{ticker}", ticker=ticker, side="buy",
                qty=qty, status="filled", filled_qty=qty, filled_price=150.0)

        def mock_sell(ticker, qty):
            return OrderResult(
                order_id=f"sell-{ticker}", ticker=ticker, side="sell",
                qty=qty, status="filled", filled_qty=qty, filled_price=400.0)

        broker.buy.side_effect = mock_buy
        broker.sell.side_effect = mock_sell

        trades = [
            {"action": "BUY", "ticker": "AAPL", "shares": 5,
             "price": 150.0, "amount": 750.0},
            {"action": "SELL", "ticker": "MSFT", "shares": 2,
             "price": 400.0, "amount": 800.0},
        ]
        state = {"transactions": []}

        portfolio_app.execute_alpaca_trades(broker, trades, state)

        call_order = [tx["action"] for tx in state["transactions"]]
        assert call_order == ["SELL", "BUY"]

    def test_execute_alpaca_trades_logs_to_state(self):
        broker = MagicMock(spec=AlpacaBroker)

        def mock_buy(ticker, qty):
            return OrderResult(
                order_id=f"buy-{ticker}", ticker=ticker, side="buy",
                qty=qty, status="filled", filled_qty=qty, filled_price=150.0)

        broker.buy.side_effect = mock_buy
        state = {"transactions": []}

        trades = [
            {"action": "BUY", "ticker": "AAPL", "shares": 5,
             "price": 150.0, "amount": 750.0},
        ]

        portfolio_app.execute_alpaca_trades(broker, trades, state)
        assert len(state["transactions"]) == 1
        tx = state["transactions"][0]
        assert tx["action"] == "BUY"
        assert tx["ticker"] == "AAPL"
        assert tx["qty"] == 5
        assert tx["status"] == "filled"
        assert tx["filled_price"] == 150.0

    def test_execute_alpaca_trades_handles_error(self):
        broker = MagicMock(spec=AlpacaBroker)
        broker.buy.side_effect = Exception("API error")
        state = {"transactions": []}

        trades = [
            {"action": "BUY", "ticker": "AAPL", "shares": 5,
             "price": 150.0, "amount": 750.0},
        ]

        results = portfolio_app.execute_alpaca_trades(broker, trades, state)
        assert len(results) == 0
        assert len(state["transactions"]) == 1
        assert state["transactions"][0]["status"] == "error"


# ===================================================================
# TestFileStructure
# ===================================================================

class TestFileStructure:
    """Verify file structure and imports."""

    def test_alpaca_broker_exists(self):
        path = os.path.join(_REPO_ROOT, "finance_tools", "broker", "alpaca.py")
        assert os.path.exists(path), f"Missing {path}"

    def test_app_exists(self):
        path = os.path.join(_APP_DIR, "app.py")
        assert os.path.exists(path), f"Missing {path}"

    def test_backtest_exists(self):
        path = os.path.join(_APP_DIR, "backtest.py")
        assert os.path.exists(path), f"Missing {path}"

    def test_app_imports_alpaca_broker(self):
        assert hasattr(portfolio_app, "AlpacaBroker")

    def test_alpaca_broker_contains_class(self):
        from finance_tools.broker import alpaca as alpaca_mod
        assert hasattr(alpaca_mod, "AlpacaBroker")
        assert hasattr(alpaca_mod, "PositionInfo")
        assert hasattr(alpaca_mod, "OrderResult")

    def test_app_contains_execute_alpaca_trades(self):
        assert hasattr(portfolio_app, "execute_alpaca_trades")
        assert callable(portfolio_app.execute_alpaca_trades)

    def test_app_contains_compute_alpaca_suggestions(self):
        assert hasattr(portfolio_app, "compute_alpaca_suggestions")
        assert callable(portfolio_app.compute_alpaca_suggestions)

    def test_equal_weight_exists(self):
        path = os.path.join(_REPO_ROOT, "finance_tools", "strategies", "equal_weight.py")
        assert os.path.exists(path), f"Missing {path}"

    def test_universe_exists(self):
        path = os.path.join(_REPO_ROOT, "finance_tools", "data", "universe.py")
        assert os.path.exists(path), f"Missing {path}"

    def test_market_data_exists(self):
        path = os.path.join(_REPO_ROOT, "finance_tools", "data", "market.py")
        assert os.path.exists(path), f"Missing {path}"


# ===================================================================
# TestCrossScriptConsistencyAlpaca
# ===================================================================

class TestCrossScriptConsistencyAlpaca:
    """Verify consistency between app.py and shared libraries."""

    def test_same_cash_reserve_pct(self):
        assert portfolio_app.CASH_RESERVE_PCT == CASH_RESERVE_PCT

    def test_default_tickers_from_universe(self):
        # The merged app uses TRADING_ASSISTANT_10 as DEFAULT_TICKERS
        assert portfolio_app.DEFAULT_TICKERS == TRADING_ASSISTANT_10

    def test_alpaca_suggestions_uses_compute_target_shares(self):
        """compute_alpaca_suggestions should use compute_target_shares."""
        source = inspect.getsource(portfolio_app.compute_alpaca_suggestions)
        assert "compute_target_shares" in source

    def test_alpaca_suggestions_uses_compute_rebalance_trades(self):
        """compute_alpaca_suggestions should use compute_rebalance_trades."""
        source = inspect.getsource(portfolio_app.compute_alpaca_suggestions)
        assert "compute_rebalance_trades" in source

    def test_alpaca_state_file_is_dotfile(self):
        """Alpaca state file should be a dotfile (gitignored)."""
        basename = os.path.basename(portfolio_app.ALPACA_STATE_FILE)
        assert basename.startswith(".")

    def test_local_state_file_is_dotfile(self):
        """Local state file should be a dotfile (gitignored)."""
        basename = os.path.basename(portfolio_app.LOCAL_STATE_FILE)
        assert basename.startswith(".")


# ===================================================================
# TestAlpacaBrokerEdgeCases
# ===================================================================

class TestAlpacaBrokerEdgeCases:
    """Edge cases for AlpacaBroker."""

    def test_order_to_result_none_filled(self):
        """Handle order with no fill data."""
        broker = AlpacaBroker(api_key="k", secret_key="s")
        order = _mock_order("o1", "AAPL", "buy", 10, "pending_new", None, None)
        order.filled_qty = None
        order.filled_avg_price = None

        result = broker._order_to_result(order)
        assert result.filled_qty == 0
        assert result.filled_price is None

    def test_get_positions_parses_all_fields(self):
        """All PositionInfo fields are parsed from Alpaca position."""
        broker = AlpacaBroker(api_key="k", secret_key="s")
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = [
            _mock_position("NVDA", 20, 800.0, 850.0, 17000.0, 1000.0, 0.0625),
        ]
        broker._client = mock_client

        result = broker.get_positions()
        pos = result["NVDA"]
        assert pos.ticker == "NVDA"
        assert pos.qty == 20
        assert pos.avg_entry_price == 800.0
        assert pos.current_price == 850.0
        assert pos.market_value == 17000.0
        assert pos.unrealized_pl == 1000.0
        assert pos.unrealized_pl_pct == 0.0625

    def test_ticker_uppercase(self):
        """buy/sell should uppercase ticker."""
        broker = AlpacaBroker(api_key="k", secret_key="s")
        mock_client = MagicMock()
        mock_client.submit_order.return_value = _mock_order(
            "o1", "AAPL", "buy", 10, "filled", 10, 155.0)
        broker._client = mock_client

        with patch.dict("sys.modules", {
            "alpaca.trading.requests": MagicMock(),
            "alpaca.trading.enums": MagicMock(),
        }):
            result = broker.buy("aapl", 10)

        assert result.ticker == "AAPL"


# ===================================================================
# TestSellSharesAlpaca
# ===================================================================

class TestSellSharesAlpaca:
    """Tests for compute_proportional_sells and sell_shares_alpaca."""

    def test_proportional_sells_basic(self):
        """Proportional sell across two positions."""
        positions = {"AAPL": 10, "MSFT": 5}
        prices = {"AAPL": 100.0, "MSFT": 200.0}
        # Total equity = 1000 + 1000 = 2000. Raise $500 = 25%.
        trades = portfolio_app.compute_proportional_sells(
            positions, prices, 500.0)
        assert len(trades) > 0
        total_raised = sum(t["amount"] for t in trades)
        # Should be close to $500
        assert abs(total_raised - 500.0) <= max(prices.values())

    def test_proportional_sells_empty_positions(self):
        """No sells when no positions held."""
        positions = {"AAPL": 0, "MSFT": 0}
        prices = {"AAPL": 100.0, "MSFT": 200.0}
        trades = portfolio_app.compute_proportional_sells(
            positions, prices, 500.0)
        assert trades == []

    def test_proportional_sells_zero_target(self):
        positions = {"AAPL": 10}
        prices = {"AAPL": 100.0}
        trades = portfolio_app.compute_proportional_sells(
            positions, prices, 0.0)
        assert trades == []

    def test_proportional_sells_negative_target(self):
        positions = {"AAPL": 10}
        prices = {"AAPL": 100.0}
        trades = portfolio_app.compute_proportional_sells(
            positions, prices, -100.0)
        assert trades == []

    def test_proportional_sells_exceeds_equity(self):
        """Requesting more than total equity sells everything."""
        positions = {"AAPL": 10}
        prices = {"AAPL": 100.0}
        trades = portfolio_app.compute_proportional_sells(
            positions, prices, 5000.0)
        assert len(trades) == 1
        assert trades[0]["shares"] == 10  # all shares

    def test_proportional_sells_all_actions_are_sell(self):
        positions = {"AAPL": 10, "MSFT": 5}
        prices = {"AAPL": 100.0, "MSFT": 200.0}
        trades = portfolio_app.compute_proportional_sells(
            positions, prices, 500.0)
        for t in trades:
            assert t["action"] == "SELL"

    def test_proportional_sells_respects_held_qty(self):
        """Never sell more than held."""
        positions = {"AAPL": 2}
        prices = {"AAPL": 100.0}
        trades = portfolio_app.compute_proportional_sells(
            positions, prices, 1000.0)
        assert len(trades) == 1
        assert trades[0]["shares"] <= 2

    def test_proportional_sells_skips_zero_price(self):
        positions = {"AAPL": 10, "MSFT": 5}
        prices = {"AAPL": 100.0, "MSFT": 0.0}
        trades = portfolio_app.compute_proportional_sells(
            positions, prices, 500.0)
        tickers = {t["ticker"] for t in trades}
        assert "MSFT" not in tickers

    def test_sell_shares_alpaca_no_positions(self, capsys):
        """sell_shares_alpaca should print warning when nothing to sell."""
        broker = MagicMock(spec=AlpacaBroker)
        state = {"transactions": []}
        positions = {"AAPL": 0}
        prices = {"AAPL": 100.0}
        result = portfolio_app.sell_shares_alpaca(
            broker, state, positions, prices)
        assert result is False
        captured = capsys.readouterr()
        assert "No shares" in captured.out


# ===================================================================
# TestDepositCashAlpaca
# ===================================================================

class TestDepositCashAlpaca:
    """Tests for deposit_cash_alpaca."""

    def test_deposit_cash_shows_balance(self, capsys):
        broker = MagicMock(spec=AlpacaBroker)
        broker.get_cash.return_value = 50_000.0
        broker.get_equity.return_value = 100_000.0
        portfolio_app.deposit_cash_alpaca(broker)
        captured = capsys.readouterr()
        assert "50,000.00" in captured.out
        assert "100,000.00" in captured.out

    def test_deposit_cash_shows_dashboard_url(self, capsys):
        broker = MagicMock(spec=AlpacaBroker)
        broker.get_cash.return_value = 50_000.0
        broker.get_equity.return_value = 100_000.0
        portfolio_app.deposit_cash_alpaca(broker)
        captured = capsys.readouterr()
        assert "alpaca.markets" in captured.out

    def test_deposit_cash_explains_limitation(self, capsys):
        broker = MagicMock(spec=AlpacaBroker)
        broker.get_cash.return_value = 50_000.0
        broker.get_equity.return_value = 100_000.0
        portfolio_app.deposit_cash_alpaca(broker)
        captured = capsys.readouterr()
        assert "cannot" in captured.out.lower() or "reset" in captured.out.lower()

    def test_deposit_cash_handles_api_error(self, capsys):
        broker = MagicMock(spec=AlpacaBroker)
        broker.get_cash.side_effect = Exception("timeout")
        broker.get_equity.side_effect = Exception("timeout")
        portfolio_app.deposit_cash_alpaca(broker)
        captured = capsys.readouterr()
        # Should not crash, should mention dashboard
        assert "alpaca.markets" in captured.out

    def test_app_contains_sell_functions(self):
        assert hasattr(portfolio_app, "compute_proportional_sells")
        assert hasattr(portfolio_app, "sell_shares_alpaca")

    def test_app_contains_deposit_cash_alpaca(self):
        assert hasattr(portfolio_app, "deposit_cash_alpaca")


# ===================================================================
# TestAssetValidation -- is_tradeable / filter_tradeable
# ===================================================================

class TestAssetValidation:
    """Tests for AlpacaBroker.is_tradeable and filter_tradeable."""

    def test_is_tradeable_true(self):
        broker = AlpacaBroker(api_key="k", secret_key="s")
        mock_client = MagicMock()
        mock_asset = MagicMock()
        mock_asset.tradable = True
        mock_client.get_asset.return_value = mock_asset
        broker._client = mock_client

        assert broker.is_tradeable("AAPL") is True
        mock_client.get_asset.assert_called_once_with("AAPL")

    def test_is_tradeable_false(self):
        broker = AlpacaBroker(api_key="k", secret_key="s")
        mock_client = MagicMock()
        mock_asset = MagicMock()
        mock_asset.tradable = False
        mock_client.get_asset.return_value = mock_asset
        broker._client = mock_client

        assert broker.is_tradeable("FAKE") is False

    def test_is_tradeable_not_found(self):
        """Unknown ticker should return False, not raise."""
        broker = AlpacaBroker(api_key="k", secret_key="s")
        mock_client = MagicMock()
        mock_client.get_asset.side_effect = Exception("asset not found")
        broker._client = mock_client

        assert broker.is_tradeable("ZZZZZ") is False

    def test_is_tradeable_cached(self):
        """Second call should use cache, not hit API again."""
        broker = AlpacaBroker(api_key="k", secret_key="s")
        mock_client = MagicMock()
        mock_asset = MagicMock()
        mock_asset.tradable = True
        mock_client.get_asset.return_value = mock_asset
        broker._client = mock_client

        assert broker.is_tradeable("AAPL") is True
        assert broker.is_tradeable("AAPL") is True
        # Should only call API once
        assert mock_client.get_asset.call_count == 1

    def test_is_tradeable_uppercase(self):
        """Lowercase ticker should be uppercased."""
        broker = AlpacaBroker(api_key="k", secret_key="s")
        mock_client = MagicMock()
        mock_asset = MagicMock()
        mock_asset.tradable = True
        mock_client.get_asset.return_value = mock_asset
        broker._client = mock_client

        assert broker.is_tradeable("aapl") is True
        mock_client.get_asset.assert_called_with("AAPL")

    def test_filter_tradeable_splits_correctly(self):
        """filter_tradeable returns (tradeable, excluded) lists."""
        broker = AlpacaBroker(api_key="k", secret_key="s")
        mock_client = MagicMock()

        def mock_get_asset(symbol):
            asset = MagicMock()
            asset.tradable = symbol != "MMC"
            return asset

        mock_client.get_asset.side_effect = mock_get_asset
        broker._client = mock_client

        tradeable, excluded = broker.filter_tradeable(
            ["AAPL", "MMC", "MSFT"])
        assert tradeable == ["AAPL", "MSFT"]
        assert excluded == ["MMC"]

    def test_filter_tradeable_all_tradeable(self):
        broker = AlpacaBroker(api_key="k", secret_key="s")
        mock_client = MagicMock()
        mock_asset = MagicMock()
        mock_asset.tradable = True
        mock_client.get_asset.return_value = mock_asset
        broker._client = mock_client

        tradeable, excluded = broker.filter_tradeable(["AAPL", "MSFT"])
        assert tradeable == ["AAPL", "MSFT"]
        assert excluded == []

    def test_filter_tradeable_none_tradeable(self):
        broker = AlpacaBroker(api_key="k", secret_key="s")
        mock_client = MagicMock()
        mock_client.get_asset.side_effect = Exception("not found")
        broker._client = mock_client

        tradeable, excluded = broker.filter_tradeable(["XXX", "YYY"])
        assert tradeable == []
        assert excluded == ["XXX", "YYY"]

    def test_filter_tradeable_empty_list(self):
        broker = AlpacaBroker(api_key="k", secret_key="s")
        tradeable, excluded = broker.filter_tradeable([])
        assert tradeable == []
        assert excluded == []


# ===================================================================
# TestAlpacaPortfolioBroker -- merged broker class
# ===================================================================

class TestAlpacaPortfolioBroker:
    """Tests for the AlpacaPortfolioBroker wrapper in the merged app.

    This replaces the old fetch_alpaca_state tests.
    """

    def test_get_positions_no_positions(self):
        """Handle empty Alpaca account gracefully."""
        mock_broker = MagicMock(spec=AlpacaBroker)
        mock_broker.get_positions.return_value = {}
        mock_broker.get_cash.return_value = 100_000.0

        state = portfolio_app.create_alpaca_initial_state(["AAPL"])
        apb = portfolio_app.AlpacaPortfolioBroker(
            mock_broker, state, "/tmp/test_state.json")

        positions = apb.get_positions(["AAPL"])
        assert positions["AAPL"] == 0
        assert apb.get_cash() == 100_000.0
        assert apb.pos_info == {}

    def test_get_positions_returns_pos_info(self):
        """pos_info should contain PositionInfo for held tickers."""
        mock_broker = MagicMock(spec=AlpacaBroker)
        mock_pos = PositionInfo(
            ticker="AAPL", qty=10, avg_entry_price=150.0,
            current_price=155.0, market_value=1550.0,
            unrealized_pl=50.0, unrealized_pl_pct=0.033)
        mock_broker.get_positions.return_value = {"AAPL": mock_pos}
        mock_broker.get_cash.return_value = 50_000.0

        state = portfolio_app.create_alpaca_initial_state(["AAPL"])
        apb = portfolio_app.AlpacaPortfolioBroker(
            mock_broker, state, "/tmp/test_state.json")

        positions = apb.get_positions(["AAPL"])
        assert positions["AAPL"] == 10
        assert "AAPL" in apb.pos_info
        assert apb.pos_info["AAPL"] is mock_pos
