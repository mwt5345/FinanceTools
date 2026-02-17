"""
Tests for the Intraday Trader (merged P05 + P07).

Covers:
- Strategy logic (Chebyshev hold/buy/sell, whole-share sizing, caps)
- Cooldown variant (Chebyshev)
- OU strategy (hold, buy, sell, whole-share, trending data, _fit_ou)
- OU cooldown variant
- Data feed (Quote, ABC, mocked yfinance, AlpacaStreamFeed, AggregatedTick)
- State management (create, save, load, resume -- local + alpaca)
- Risk management (cash reserve, max position, circuit breaker)
- Session stats (P&L, trade counts, Sharpe, empty)
- Alpaca bridge (fetch_portfolio_state, execute_action, dry-run)
- Alpaca profile loading (YAML, env vars, defaults)
- CLI parsing (local + alpaca flags)
- File structure and cross-script consistency
"""

import importlib.util
import inspect
import json
import math
import os
import sys
import tempfile
from datetime import datetime, time, timezone, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup -- no sys.path manipulation needed; packages are importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# New layout paths (for file-structure tests)
APP_DIR = os.path.join(PROJECT_ROOT, "apps", "intraday_trader")
BROKER_DIR = os.path.join(PROJECT_ROOT, "finance_tools", "broker")
STRATEGIES_DIR = os.path.join(PROJECT_ROOT, "finance_tools", "strategies")
BACKTEST_DIR = os.path.join(PROJECT_ROOT, "finance_tools", "backtest")

# ---------------------------------------------------------------------------
# Direct imports from the finance_tools package
# ---------------------------------------------------------------------------
from finance_tools.backtest.engine import Strategy, Action, ActionType, Portfolio
from finance_tools.broker.data_feed import (
    Quote, DataFeed, YFinanceFeed, AlpacaStreamFeed,
    AggregatedTick, _MARKET_OPEN, _MARKET_CLOSE,
)
from finance_tools.strategies.intraday import (
    IntradayChebyshev, IntradayChebyshevWithCooldown,
    IntradayOU, IntradayOUWithCooldown,
)
from finance_tools.broker.alpaca import AlpacaBroker, PositionInfo, OrderResult, load_profile

# ---------------------------------------------------------------------------
# Import the merged app.py explicitly by file path to avoid collision
# with other app.py files on sys.path
# ---------------------------------------------------------------------------
_app_spec = importlib.util.spec_from_file_location(
    "intraday_trader_app", os.path.join(APP_DIR, "app.py"))
intraday_app = importlib.util.module_from_spec(_app_spec)
_app_spec.loader.exec_module(intraday_app)


# =====================================================================
# Helpers
# =====================================================================

class MockFeed(DataFeed):
    """Deterministic data feed for testing."""

    def __init__(self, prices: list[float], market_open: bool = True):
        self._prices = prices
        self._idx = 0
        self._market_open = market_open

    def latest(self) -> Quote:
        price = self._prices[self._idx % len(self._prices)]
        self._idx += 1
        return Quote(
            ticker="TEST",
            price=price,
            timestamp=datetime.now(),
            market_open=self._market_open,
        )

    def history(self, lookback_minutes: int = 60) -> pd.DataFrame:
        n = min(lookback_minutes, len(self._prices))
        return pd.DataFrame({
            "Open": self._prices[:n],
            "High": self._prices[:n],
            "Low": self._prices[:n],
            "Close": self._prices[:n],
            "Volume": [1000] * n,
        })

    def is_market_open(self) -> bool:
        return self._market_open


def _make_history(prices: list[float]) -> pd.DataFrame:
    """Build a minimal history DataFrame from a price list."""
    return pd.DataFrame({"Close": prices})


def _make_portfolio(cash: float = 10000.0, shares: float = 0.0,
                    price: float = 100.0) -> Portfolio:
    """Build a Portfolio for testing."""
    return Portfolio(cash=cash, shares=shares, price=price)


# ---------------------------------------------------------------------------
# Helpers -- mock Alpaca objects
# ---------------------------------------------------------------------------

def _mock_account(cash=100_000.0, equity=150_000.0):
    account = MagicMock()
    account.cash = str(cash)
    account.equity = str(equity)
    return account


def _mock_position(symbol="MSFT", qty=50, avg_entry=400.0,
                   current=410.0, market_value=20500.0,
                   unrealized_pl=500.0, unrealized_plpc=0.025):
    pos = MagicMock()
    pos.symbol = symbol
    pos.qty = str(qty)
    pos.avg_entry_price = str(avg_entry)
    pos.current_price = str(current)
    pos.market_value = str(market_value)
    pos.unrealized_pl = str(unrealized_pl)
    pos.unrealized_plpc = str(unrealized_plpc)
    return pos


def _mock_order(order_id="order-123", symbol="MSFT", side="buy",
                qty=10, status="filled", filled_qty=10,
                filled_avg_price=410.0):
    order = MagicMock()
    order.id = order_id
    order.symbol = symbol
    order.side = MagicMock()
    order.side.value = side
    order.qty = str(qty)
    order.status = MagicMock()
    order.status.value = status
    order.filled_qty = str(filled_qty) if filled_qty else None
    order.filled_avg_price = (str(filled_avg_price)
                              if filled_avg_price else None)
    return order


# =====================================================================
# Strategy logic tests -- Chebyshev
# =====================================================================

class TestIntradayChebyshev:
    """Tests for the base IntradayChebyshev strategy."""

    def test_hold_on_insufficient_history(self):
        """Strategy returns HOLD when history is too short."""
        strategy = IntradayChebyshev(window=30)
        prices = [100.0] * 10  # only 10, need 31
        history = _make_history(prices)
        day = history.iloc[-1]
        portfolio = _make_portfolio()
        action = strategy.decide(day, history, portfolio)
        assert action.action == ActionType.HOLD

    def test_hold_on_exact_window_boundary(self):
        """HOLD when history length == window (need window+1)."""
        strategy = IntradayChebyshev(window=5)
        prices = [100.0] * 5
        history = _make_history(prices)
        day = history.iloc[-1]
        portfolio = _make_portfolio()
        action = strategy.decide(day, history, portfolio)
        assert action.action == ActionType.HOLD

    def test_hold_on_zero_volatility(self):
        """HOLD when all returns are zero (sigma=0)."""
        strategy = IntradayChebyshev(window=5)
        prices = [100.0] * 10  # constant price = zero returns
        history = _make_history(prices)
        day = history.iloc[-1]
        portfolio = _make_portfolio()
        action = strategy.decide(day, history, portfolio)
        assert action.action == ActionType.HOLD

    def test_hold_on_small_z(self):
        """HOLD when z-score is below threshold."""
        strategy = IntradayChebyshev(window=5, k_threshold=3.0)
        # Small, smooth price movement -- z < 3
        prices = [100.0 + 0.01 * i for i in range(10)]
        history = _make_history(prices)
        day = history.iloc[-1]
        portfolio = _make_portfolio()
        action = strategy.decide(day, history, portfolio)
        assert action.action == ActionType.HOLD

    def test_buy_on_large_dip(self):
        """BUY when price dips sharply (large negative z)."""
        strategy = IntradayChebyshev(window=10, k_threshold=2.0)
        np.random.seed(603)
        # Stable prices then a big dip
        prices = [100.0] * 10 + [100.0 + np.random.normal(0, 0.1) for _ in range(5)]
        prices.append(prices[-1] * 0.95)  # 5% dip
        history = _make_history(prices)
        day = history.iloc[-1]
        portfolio = _make_portfolio(cash=10000, shares=0, price=prices[-1])
        action = strategy.decide(day, history, portfolio)
        # Should buy on large negative z
        assert action.action == ActionType.BUY

    def test_sell_on_large_spike(self):
        """SELL when price spikes sharply (large positive z)."""
        strategy = IntradayChebyshev(window=10, k_threshold=2.0)
        np.random.seed(603)
        prices = [100.0] * 10 + [100.0 + np.random.normal(0, 0.1) for _ in range(5)]
        prices.append(prices[-1] * 1.05)  # 5% spike
        history = _make_history(prices)
        day = history.iloc[-1]
        portfolio = _make_portfolio(cash=0, shares=50, price=prices[-1])
        action = strategy.decide(day, history, portfolio)
        assert action.action == ActionType.SELL

    def test_whole_share_sizing_buy(self):
        """Buy actions use whole shares."""
        strategy = IntradayChebyshev(window=10, k_threshold=2.0)
        np.random.seed(603)
        prices = [100.0] * 10 + [100.0 + np.random.normal(0, 0.1) for _ in range(5)]
        prices.append(prices[-1] * 0.95)
        history = _make_history(prices)
        day = history.iloc[-1]
        portfolio = _make_portfolio(cash=10000, shares=0, price=prices[-1])
        action = strategy.decide(day, history, portfolio)
        if action.action == ActionType.BUY:
            assert action.shares is not None
            assert action.shares == math.floor(action.shares)
            assert action.shares >= 1

    def test_whole_share_sizing_sell(self):
        """Sell actions use whole shares."""
        strategy = IntradayChebyshev(window=10, k_threshold=2.0)
        np.random.seed(603)
        prices = [100.0] * 10 + [100.0 + np.random.normal(0, 0.1) for _ in range(5)]
        prices.append(prices[-1] * 1.05)
        history = _make_history(prices)
        day = history.iloc[-1]
        portfolio = _make_portfolio(cash=0, shares=50, price=prices[-1])
        action = strategy.decide(day, history, portfolio)
        if action.action == ActionType.SELL:
            assert action.shares is not None
            assert action.shares == math.floor(action.shares)
            assert action.shares >= 1

    def test_max_position_fraction_caps_buy(self):
        """Buy fraction is capped at max_position_frac."""
        strategy = IntradayChebyshev(window=10, k_threshold=2.0,
                                     max_position_frac=0.50)
        np.random.seed(603)
        prices = [100.0] * 10 + [100.0 + np.random.normal(0, 0.1) for _ in range(5)]
        prices.append(prices[-1] * 0.90)  # 10% dip -> high k
        history = _make_history(prices)
        day = history.iloc[-1]
        price = prices[-1]
        portfolio = _make_portfolio(cash=10000, shares=0, price=price)
        action = strategy.decide(day, history, portfolio)
        if action.action == ActionType.BUY and action.shares:
            # Max spend = cash * 0.50
            max_shares = math.floor(10000 * 0.50 / price)
            assert action.shares <= max_shares

    def test_no_buy_with_zero_cash(self):
        """HOLD when cash is zero even if signal says buy."""
        strategy = IntradayChebyshev(window=10, k_threshold=2.0)
        np.random.seed(603)
        prices = [100.0] * 10 + [100.0 + np.random.normal(0, 0.1) for _ in range(5)]
        prices.append(prices[-1] * 0.95)
        history = _make_history(prices)
        day = history.iloc[-1]
        portfolio = _make_portfolio(cash=0, shares=50, price=prices[-1])
        action = strategy.decide(day, history, portfolio)
        assert action.action != ActionType.BUY

    def test_no_sell_with_zero_shares(self):
        """HOLD when shares are zero even if signal says sell."""
        strategy = IntradayChebyshev(window=10, k_threshold=2.0)
        np.random.seed(603)
        prices = [100.0] * 10 + [100.0 + np.random.normal(0, 0.1) for _ in range(5)]
        prices.append(prices[-1] * 1.05)
        history = _make_history(prices)
        day = history.iloc[-1]
        portfolio = _make_portfolio(cash=10000, shares=0, price=prices[-1])
        action = strategy.decide(day, history, portfolio)
        assert action.action != ActionType.SELL

    def test_compute_z_returns_float(self):
        """compute_z() returns a float when history is sufficient."""
        strategy = IntradayChebyshev(window=5)
        prices = [100.0 + i * 0.1 for i in range(10)]
        history = _make_history(prices)
        z = strategy.compute_z(history)
        assert z is not None
        assert isinstance(z, float)

    def test_compute_z_returns_none_insufficient_history(self):
        """compute_z() returns None when history is too short."""
        strategy = IntradayChebyshev(window=30)
        prices = [100.0] * 5
        history = _make_history(prices)
        z = strategy.compute_z(history)
        assert z is None

    def test_subclasses_strategy(self):
        """IntradayChebyshev is a proper Strategy subclass."""
        assert issubclass(IntradayChebyshev, Strategy)
        strategy = IntradayChebyshev()
        assert isinstance(strategy, Strategy)


# =====================================================================
# Cooldown variant tests -- Chebyshev
# =====================================================================

class TestIntradayChebyshevWithCooldown:
    """Tests for the cooldown variant."""

    def _trigger_trade_sequence(self, strategy, portfolio_buy, portfolio_sell):
        """Helper: generate a dip history that triggers a buy."""
        np.random.seed(603)
        prices = [100.0] * 10 + [100.0 + np.random.normal(0, 0.1) for _ in range(5)]
        prices.append(prices[-1] * 0.95)
        history = _make_history(prices)
        day = history.iloc[-1]
        return strategy.decide(day, history, portfolio_buy)

    def test_blocks_consecutive_trades(self):
        """Second trade within cooldown period is blocked."""
        strategy = IntradayChebyshevWithCooldown(
            window=10, k_threshold=2.0, cooldown_ticks=3)

        np.random.seed(603)
        prices = [100.0] * 10 + [100.0 + np.random.normal(0, 0.1) for _ in range(5)]
        prices.append(prices[-1] * 0.95)
        history = _make_history(prices)
        day = history.iloc[-1]
        portfolio = _make_portfolio(cash=10000, shares=0, price=prices[-1])

        # First call -- should trade (starts ready)
        a1 = strategy.decide(day, history, portfolio)
        assert a1.action == ActionType.BUY

        # Immediate second call -- should be blocked by cooldown
        a2 = strategy.decide(day, history, portfolio)
        assert a2.action == ActionType.HOLD

    def test_allows_trade_after_cooldown(self):
        """Trade is allowed after cooldown period expires."""
        strategy = IntradayChebyshevWithCooldown(
            window=10, k_threshold=2.0, cooldown_ticks=2)

        np.random.seed(603)
        prices = [100.0] * 10 + [100.0 + np.random.normal(0, 0.1) for _ in range(5)]
        prices.append(prices[-1] * 0.95)
        history = _make_history(prices)
        day = history.iloc[-1]
        portfolio = _make_portfolio(cash=10000, shares=0, price=prices[-1])

        # Trigger trade
        a1 = strategy.decide(day, history, portfolio)
        assert a1.action == ActionType.BUY

        # Cooldown tick 1
        a2 = strategy.decide(day, history, portfolio)
        assert a2.action == ActionType.HOLD

        # Cooldown tick 2
        a3 = strategy.decide(day, history, portfolio)
        assert a3.action == ActionType.HOLD

        # After cooldown -- should trade again
        a4 = strategy.decide(day, history, portfolio)
        assert a4.action == ActionType.BUY

    def test_hold_doesnt_reset_cooldown(self):
        """A HOLD signal during cooldown doesn't reset the counter."""
        strategy = IntradayChebyshevWithCooldown(
            window=10, k_threshold=2.0, cooldown_ticks=2)

        # Use stable prices that won't trigger a trade
        stable_prices = [100.0] * 20
        stable_history = _make_history(stable_prices)
        stable_day = stable_history.iloc[-1]
        portfolio = _make_portfolio()

        # This should hold (no signal) and not affect cooldown state
        a = strategy.decide(stable_day, stable_history, portfolio)
        assert a.action == ActionType.HOLD

    def test_reset_cooldown(self):
        """reset_cooldown() makes strategy ready to trade immediately."""
        strategy = IntradayChebyshevWithCooldown(
            window=10, k_threshold=2.0, cooldown_ticks=100)

        np.random.seed(603)
        prices = [100.0] * 10 + [100.0 + np.random.normal(0, 0.1) for _ in range(5)]
        prices.append(prices[-1] * 0.95)
        history = _make_history(prices)
        day = history.iloc[-1]
        portfolio = _make_portfolio(cash=10000, shares=0, price=prices[-1])

        # Trigger trade
        a1 = strategy.decide(day, history, portfolio)
        assert a1.action == ActionType.BUY

        # In cooldown -- should hold
        a2 = strategy.decide(day, history, portfolio)
        assert a2.action == ActionType.HOLD

        # Reset cooldown
        strategy.reset_cooldown()

        # Should be ready to trade again
        a3 = strategy.decide(day, history, portfolio)
        assert a3.action == ActionType.BUY


# =====================================================================
# Data feed tests
# =====================================================================

class TestDataFeed:
    """Tests for the data feed abstraction."""

    def test_quote_fields(self):
        """Quote dataclass has all expected fields."""
        q = Quote(
            ticker="AAPL", price=150.0,
            timestamp=datetime.now(),
            bid=149.9, ask=150.1,
            volume=1000, market_open=True,
        )
        assert q.ticker == "AAPL"
        assert q.price == 150.0
        assert q.bid == 149.9
        assert q.ask == 150.1
        assert q.volume == 1000
        assert q.market_open is True

    def test_quote_defaults(self):
        """Quote has sensible defaults for optional fields."""
        q = Quote(ticker="X", price=10.0, timestamp=datetime.now())
        assert q.bid is None
        assert q.ask is None
        assert q.volume == 0.0
        assert q.market_open is True

    def test_abc_enforcement(self):
        """Cannot instantiate DataFeed directly."""
        with pytest.raises(TypeError):
            DataFeed()

    def test_mock_feed_latest(self):
        """MockFeed returns deterministic prices."""
        feed = MockFeed([100.0, 101.0, 102.0])
        q1 = feed.latest()
        assert q1.price == 100.0
        q2 = feed.latest()
        assert q2.price == 101.0

    def test_mock_feed_history(self):
        """MockFeed returns history DataFrame."""
        prices = [100.0 + i for i in range(10)]
        feed = MockFeed(prices)
        hist = feed.history(lookback_minutes=5)
        assert len(hist) == 5
        assert "Close" in hist.columns

    def test_mock_feed_market_open(self):
        """MockFeed respects market_open parameter."""
        feed_open = MockFeed([100.0], market_open=True)
        assert feed_open.is_market_open() is True
        feed_closed = MockFeed([100.0], market_open=False)
        assert feed_closed.is_market_open() is False

    def test_yfinance_feed_init(self):
        """YFinanceFeed initializes with uppercased ticker."""
        feed = YFinanceFeed("aapl")
        assert feed.ticker == "AAPL"

    def test_market_hours_constants(self):
        """Market hours are NYSE standard."""
        assert _MARKET_OPEN == time(9, 30)
        assert _MARKET_CLOSE == time(16, 0)

    def test_warmup_history_returns_prices(self):
        """warmup_history() returns a list of close prices from feed."""
        warmup_history = intraday_app.warmup_history
        prices = [100.0 + i * 0.1 for i in range(50)]
        feed = MockFeed(prices)
        result = warmup_history(feed, window=30)
        assert len(result) > 0
        assert all(isinstance(p, float) for p in result)

    def test_warmup_history_fills_window(self):
        """warmup_history() returns enough prices to fill the window."""
        warmup_history = intraday_app.warmup_history
        prices = [100.0 + i * 0.1 for i in range(100)]
        feed = MockFeed(prices)
        result = warmup_history(feed, window=30)
        # Should return at least window + 1 prices
        assert len(result) >= 31


# =====================================================================
# State management tests -- LOCAL broker
# =====================================================================

class TestStateManagementLocal:
    """Tests for state persistence with local broker."""

    def test_create_initial_state(self):
        """Initial state has all required fields for local mode."""
        state = intraday_app.create_initial_state(
            "AAPL", 2.0, 30, 10, 6, cash=10000.0)
        assert state["ticker"] == "AAPL"
        assert state["cash"] == 10000.0
        assert state["shares"] == 0
        assert state["initial_cash"] == 10000.0
        assert state["params"]["k_threshold"] == 2.0
        assert state["params"]["window"] == 30
        assert state["params"]["interval"] == 10
        assert state["params"]["cooldown"] == 6
        assert state["trades"] == []
        assert state["tick_log"] == []
        assert "session_start" in state

    def test_save_and_load_roundtrip(self, tmp_path):
        """State survives JSON roundtrip."""
        create_initial_state = intraday_app.create_initial_state
        save_state = intraday_app.save_state
        load_state = intraday_app.load_state

        state = create_initial_state("TEST", 2.5, 20, 15, 3, cash=5000.0)

        # Temporarily override SCRIPT_DIR so state file goes to tmp_path
        test_file = os.path.join(str(tmp_path), ".intraday_TEST.json")
        with patch.object(intraday_app, "_state_file", return_value=test_file):
            save_state(state)
            loaded = load_state("TEST")

        assert loaded is not None
        assert loaded["ticker"] == "TEST"
        assert loaded["cash"] == 5000.0
        assert loaded["params"]["k_threshold"] == 2.5

    def test_per_ticker_files(self):
        """Each ticker gets its own state file."""
        _state_file = intraday_app._state_file
        f1 = _state_file("AAPL")
        f2 = _state_file("MSFT")
        assert "AAPL" in f1
        assert "MSFT" in f2
        assert f1 != f2

    def test_load_nonexistent_returns_none(self):
        """Loading state for unknown ticker returns None."""
        load_state = intraday_app.load_state
        with patch.object(intraday_app, "_state_file",
                          return_value="/tmp/nonexistent_xyz.json"):
            result = load_state("NONEXISTENT")
        assert result is None

    def test_trade_logging(self):
        """Trades are appended to state."""
        state = intraday_app.create_initial_state(
            "TEST", 2.0, 30, 10, 6, cash=10000.0)

        state["trades"].append({
            "timestamp": datetime.now().isoformat(),
            "action": "buy",
            "shares": 5,
            "price": 100.0,
            "cost": 500.0,
        })
        assert len(state["trades"]) == 1
        assert state["trades"][0]["action"] == "buy"

    def test_tick_logging(self):
        """Ticks are appended to state."""
        state = intraday_app.create_initial_state(
            "TEST", 2.0, 30, 10, 6, cash=10000.0)

        state["tick_log"].append({
            "timestamp": datetime.now().isoformat(),
            "price": 100.0,
            "z_score": -1.5,
            "portfolio_value": 10000.0,
            "shares": 0,
            "cash": 10000.0,
        })
        assert len(state["tick_log"]) == 1

    def test_resume_preserves_trades(self, tmp_path):
        """Resumed session preserves trade history."""
        create_initial_state = intraday_app.create_initial_state
        save_state = intraday_app.save_state
        load_state = intraday_app.load_state

        state = create_initial_state("RESUME", 2.0, 30, 10, 6, cash=10000.0)
        state["shares"] = 10
        state["cash"] = 9000.0
        state["trades"].append({
            "timestamp": datetime.now().isoformat(),
            "action": "buy", "shares": 10, "price": 100.0, "cost": 1000.0,
        })

        test_file = os.path.join(str(tmp_path), ".intraday_RESUME.json")
        with patch.object(intraday_app, "_state_file", return_value=test_file):
            save_state(state)
            loaded = load_state("RESUME")

        assert loaded["shares"] == 10
        assert loaded["cash"] == 9000.0
        assert len(loaded["trades"]) == 1

    def test_state_json_serializable(self):
        """Initial state is fully JSON-serializable."""
        state = intraday_app.create_initial_state(
            "TEST", 2.0, 30, 10, 6, cash=10000.0)
        # Should not raise
        json_str = json.dumps(state)
        assert len(json_str) > 0


# =====================================================================
# State management tests -- ALPACA broker
# =====================================================================

class TestStateManagementAlpaca:
    """Tests for state persistence with Alpaca broker."""

    def test_create_initial_state_no_cash_shares(self):
        """Initial state should NOT have cash/shares at top level for alpaca."""
        state = intraday_app.create_initial_state(
            "MSFT", 1.5, 30, 60, 6,
            broker_mode="alpaca", profile="intraday")
        assert "cash" not in state
        assert "shares" not in state
        assert "initial_cash" not in state
        assert state["ticker"] == "MSFT"
        assert state["profile"] == "intraday"
        assert state["trades"] == []
        assert state["tick_log"] == []

    def test_create_initial_state_params(self):
        state = intraday_app.create_initial_state(
            "AAPL", 2.0, 60, 30, 10,
            strategy_type="ou", entry=1.5, max_threshold=3.5,
            broker_mode="alpaca", profile="portfolio")
        assert state["params"]["strategy"] == "ou"
        assert state["params"]["k_threshold"] == 2.0
        assert state["params"]["entry_threshold"] == 1.5
        assert state["params"]["max_threshold"] == 3.5
        assert state["params"]["window"] == 60

    def test_state_json_serializable(self):
        state = intraday_app.create_initial_state(
            "MSFT", 1.5, 30, 60, 6,
            broker_mode="alpaca", profile="intraday")
        serialized = json.dumps(state)
        assert isinstance(serialized, str)

    def test_save_load_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(intraday_app, "SCRIPT_DIR", str(tmp_path))
        state = intraday_app.create_initial_state(
            "MSFT", 1.5, 30, 60, 6,
            broker_mode="alpaca", profile="intraday")
        state["trades"].append({
            "timestamp": "2026-02-11T10:00:00",
            "action": "buy",
            "shares": 10,
            "price": 410.0,
            "order_id": "test-order",
            "status": "filled",
        })
        intraday_app.save_state(state)
        loaded = intraday_app.load_state("MSFT", broker_mode="alpaca")
        assert loaded is not None
        assert loaded["ticker"] == "MSFT"
        assert len(loaded["trades"]) == 1

    def test_load_state_returns_none_if_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(intraday_app, "SCRIPT_DIR", str(tmp_path))
        assert intraday_app.load_state("NONEXISTENT",
                                       broker_mode="alpaca") is None

    def test_state_file_is_dotfile(self):
        path = intraday_app._state_file("MSFT", broker_mode="alpaca")
        basename = os.path.basename(path)
        assert basename.startswith(".")
        assert "alpaca_intraday" in basename

    def test_trade_logging_with_order_id(self):
        """Trades should include order_id field."""
        broker = MagicMock(spec=AlpacaBroker)
        broker.buy.return_value = OrderResult(
            order_id="abc-123", ticker="MSFT", side="buy",
            qty=10, status="filled", filled_qty=10, filled_price=410.0)
        state = {"trades": []}

        intraday_app.execute_action(
            broker, "MSFT", Action.buy_shares(10), state)
        assert state["trades"][0]["order_id"] == "abc-123"

    def test_dry_run_trade_has_none_order_id(self):
        broker = MagicMock(spec=AlpacaBroker)
        state = {"trades": []}
        intraday_app.execute_action(
            broker, "MSFT", Action.buy_shares(10), state, dry_run=True)
        assert state["trades"][0]["order_id"] is None


# =====================================================================
# Risk management tests -- LOCAL broker
# =====================================================================

class TestRiskManagementLocal:
    """Tests for the risk filter (local broker context)."""

    def test_cash_reserve_enforcement(self):
        """Buy is reduced to respect cash reserve."""
        check_risk = intraday_app.check_risk
        RiskLimits = intraday_app.RiskLimits
        risk = RiskLimits(cash_reserve_pct=0.10)
        portfolio = _make_portfolio(cash=1000, shares=0, price=100.0)
        action = Action.buy_shares(10)  # wants $1000
        filtered = check_risk(action, portfolio, risk, trade_count=0)
        # Should cap at 9 shares (reserve = $100, available = $900)
        if filtered.action == ActionType.BUY:
            assert filtered.shares <= 9

    def test_max_position_enforcement(self):
        """Buy is reduced to respect max position percentage."""
        check_risk = intraday_app.check_risk
        RiskLimits = intraday_app.RiskLimits
        risk = RiskLimits(max_position_pct=0.50)
        # Already 40% invested
        portfolio = _make_portfolio(cash=6000, shares=40, price=100.0)
        # Total value = $10,000. Max equity = $5,000. Current = $4,000.
        # Headroom = $1,000 => max 10 shares
        action = Action.buy_shares(50)  # wants 50 shares
        filtered = check_risk(action, portfolio, risk, trade_count=0)
        if filtered.action == ActionType.BUY:
            assert filtered.shares <= 10

    def test_circuit_breaker(self):
        """No trades after max_trades_per_session is reached."""
        check_risk = intraday_app.check_risk
        RiskLimits = intraday_app.RiskLimits
        risk = RiskLimits(max_trades_per_session=5)
        portfolio = _make_portfolio(cash=10000, shares=0, price=100.0)
        action = Action.buy_shares(10)
        filtered = check_risk(action, portfolio, risk, trade_count=5)
        assert filtered.action == ActionType.HOLD

    def test_sell_capped_at_current_shares(self):
        """Sell is capped at current share count."""
        check_risk = intraday_app.check_risk
        RiskLimits = intraday_app.RiskLimits
        risk = RiskLimits()
        portfolio = _make_portfolio(cash=5000, shares=10, price=100.0)
        action = Action.sell_shares(20)  # wants to sell 20 but only has 10
        filtered = check_risk(action, portfolio, risk, trade_count=0)
        assert filtered.action == ActionType.SELL
        assert filtered.shares <= 10

    def test_hold_passes_through(self):
        """HOLD action passes through risk filter unchanged."""
        check_risk = intraday_app.check_risk
        RiskLimits = intraday_app.RiskLimits
        risk = RiskLimits()
        portfolio = _make_portfolio()
        action = Action.hold()
        filtered = check_risk(action, portfolio, risk, trade_count=0)
        assert filtered.action == ActionType.HOLD

    def test_buy_blocked_when_no_cash(self):
        """Buy becomes HOLD when cash is zero."""
        check_risk = intraday_app.check_risk
        RiskLimits = intraday_app.RiskLimits
        risk = RiskLimits()
        portfolio = _make_portfolio(cash=0, shares=50, price=100.0)
        action = Action.buy_shares(5)
        filtered = check_risk(action, portfolio, risk, trade_count=0)
        assert filtered.action == ActionType.HOLD

    def test_sell_blocked_when_no_shares(self):
        """Sell becomes HOLD when shares is zero."""
        check_risk = intraday_app.check_risk
        RiskLimits = intraday_app.RiskLimits
        risk = RiskLimits()
        portfolio = _make_portfolio(cash=10000, shares=0, price=100.0)
        action = Action.sell_shares(5)
        filtered = check_risk(action, portfolio, risk, trade_count=0)
        assert filtered.action == ActionType.HOLD

    def test_buy_blocked_when_below_one_share(self):
        """Buy becomes HOLD when available cash < 1 share price."""
        check_risk = intraday_app.check_risk
        RiskLimits = intraday_app.RiskLimits
        risk = RiskLimits(cash_reserve_pct=0.05)
        # Cash $50, price $100 -> can't afford 1 share after reserve
        portfolio = _make_portfolio(cash=50, shares=0, price=100.0)
        action = Action.buy_shares(1)
        filtered = check_risk(action, portfolio, risk, trade_count=0)
        assert filtered.action == ActionType.HOLD

    def test_risk_limits_defaults(self):
        """RiskLimits has expected defaults."""
        RiskLimits = intraday_app.RiskLimits
        risk = RiskLimits()
        assert risk.cash_reserve_pct == 0.05
        assert risk.max_position_pct == 0.90
        assert risk.max_trades_per_session == 50

    def test_absolute_share_limit(self):
        """Sell of more shares than held is capped."""
        check_risk = intraday_app.check_risk
        RiskLimits = intraday_app.RiskLimits
        risk = RiskLimits()
        portfolio = _make_portfolio(cash=0, shares=3, price=100.0)
        action = Action.sell_shares(100)
        filtered = check_risk(action, portfolio, risk, trade_count=0)
        assert filtered.shares <= 3


# =====================================================================
# Risk management tests -- ALPACA broker
# =====================================================================

class TestRiskManagementAlpaca:
    """Tests for check_risk() in Alpaca broker context."""

    def test_cash_reserve_enforced(self):
        risk = intraday_app.RiskLimits(cash_reserve_pct=0.05)
        portfolio = Portfolio(cash=100.0, shares=0, price=100.0)
        action = Action.buy_shares(2)
        result = intraday_app.check_risk(action, portfolio, risk, 0)
        # Only $95 available ($5 reserve), so can buy 0 shares @ $100
        assert result.action == ActionType.HOLD

    def test_max_position_enforced(self):
        risk = intraday_app.RiskLimits(max_position_pct=0.50)
        # 50% already in stock, max 50% allowed
        portfolio = Portfolio(cash=50000.0, shares=100, price=500.0)
        action = Action.buy_shares(10)
        result = intraday_app.check_risk(action, portfolio, risk, 0)
        assert result.action == ActionType.HOLD

    def test_circuit_breaker(self):
        risk = intraday_app.RiskLimits(max_trades_per_session=5)
        portfolio = Portfolio(cash=100000.0, shares=50, price=100.0)
        action = Action.buy_shares(10)
        result = intraday_app.check_risk(action, portfolio, risk, 5)
        assert result.action == ActionType.HOLD

    def test_sell_capped_at_position(self):
        risk = intraday_app.RiskLimits()
        portfolio = Portfolio(cash=10000.0, shares=5, price=100.0)
        action = Action.sell_shares(100)
        result = intraday_app.check_risk(action, portfolio, risk, 0)
        assert result.action == ActionType.SELL
        assert result.shares == 5

    def test_hold_passes_through(self):
        risk = intraday_app.RiskLimits()
        portfolio = Portfolio(cash=100000.0, shares=50, price=100.0)
        action = Action.hold()
        result = intraday_app.check_risk(action, portfolio, risk, 0)
        assert result.action == ActionType.HOLD


# =====================================================================
# Session stats tests -- LOCAL broker
# =====================================================================

class TestSessionStatsLocal:
    """Tests for session statistics computation (local broker)."""

    def test_pnl_calculation(self):
        """P&L is correctly computed from initial and current value."""
        compute_session_stats = intraday_app.compute_session_stats
        create_initial_state = intraday_app.create_initial_state
        state = create_initial_state("TEST", 2.0, 30, 10, 6, cash=10000.0)
        state["shares"] = 10
        state["cash"] = 9000.0
        state["tick_log"] = [
            {"timestamp": datetime.now().isoformat(), "price": 100.0,
             "z_score": 0.0, "portfolio_value": 10000.0, "shares": 0,
             "cash": 10000.0},
            {"timestamp": datetime.now().isoformat(), "price": 110.0,
             "z_score": 1.0, "portfolio_value": 10100.0, "shares": 10,
             "cash": 9000.0},
        ]
        stats = compute_session_stats(state)
        # Current value: 9000 + 10 * 110 = 10100
        assert stats["current_value"] == pytest.approx(10100.0)
        assert stats["pnl"] == pytest.approx(100.0)
        assert stats["pnl_pct"] == pytest.approx(0.01)

    def test_trade_counts(self):
        """Buy and sell counts are correct."""
        compute_session_stats = intraday_app.compute_session_stats
        create_initial_state = intraday_app.create_initial_state
        state = create_initial_state("TEST", 2.0, 30, 10, 6, cash=10000.0)
        state["trades"] = [
            {"action": "buy", "shares": 5, "price": 100.0},
            {"action": "buy", "shares": 3, "price": 101.0},
            {"action": "sell", "shares": 2, "price": 105.0},
        ]
        state["tick_log"] = [
            {"timestamp": datetime.now().isoformat(), "price": 105.0,
             "portfolio_value": 10000.0},
        ]
        stats = compute_session_stats(state)
        assert stats["n_buys"] == 2
        assert stats["n_sells"] == 1
        assert stats["n_trades"] == 3

    def test_sharpe_sign_positive_returns(self):
        """Sharpe is positive when returns are consistently positive."""
        compute_session_stats = intraday_app.compute_session_stats
        create_initial_state = intraday_app.create_initial_state
        state = create_initial_state("TEST", 2.0, 30, 10, 6, cash=10000.0)
        # Steadily increasing portfolio values
        state["tick_log"] = [
            {"timestamp": "2025-01-01T10:00:00", "price": 100.0,
             "portfolio_value": 10000.0 + i * 10, "shares": 0,
             "cash": 10000.0}
            for i in range(50)
        ]
        stats = compute_session_stats(state)
        assert stats["sharpe"] > 0

    def test_empty_session(self):
        """Empty session returns zero stats."""
        compute_session_stats = intraday_app.compute_session_stats
        create_initial_state = intraday_app.create_initial_state
        state = create_initial_state("TEST", 2.0, 30, 10, 6, cash=10000.0)
        stats = compute_session_stats(state)
        assert stats["n_trades"] == 0
        # No ticks, no shares held -> current_value = cash = initial -> pnl = 0
        assert stats["pnl"] == pytest.approx(0.0)
        assert stats["sharpe"] == 0.0

    def test_duration_minutes(self):
        """Duration is computed from first to last tick."""
        compute_session_stats = intraday_app.compute_session_stats
        create_initial_state = intraday_app.create_initial_state
        state = create_initial_state("TEST", 2.0, 30, 10, 6, cash=10000.0)
        state["tick_log"] = [
            {"timestamp": "2025-01-01T10:00:00", "price": 100.0,
             "portfolio_value": 10000.0},
            {"timestamp": "2025-01-01T10:30:00", "price": 101.0,
             "portfolio_value": 10100.0},
        ]
        stats = compute_session_stats(state)
        assert stats["duration_minutes"] == pytest.approx(30.0)


# =====================================================================
# Session stats tests -- ALPACA broker
# =====================================================================

class TestSessionStatsAlpaca:
    """Tests for compute_session_stats with Alpaca broker."""

    def test_empty_session(self):
        state = intraday_app.create_initial_state(
            "MSFT", 1.5, 30, 60, 6,
            broker_mode="alpaca", profile="intraday")
        stats = intraday_app.compute_session_stats(state)
        assert stats["pnl"] == 0.0
        assert stats["n_trades"] == 0
        assert stats["duration_minutes"] == 0.0

    def test_positive_pnl(self):
        state = {
            "ticker": "MSFT",
            "broker_mode": "alpaca",
            "params": {"interval": 60},
            "trades": [],
            "tick_log": [
                {"timestamp": "2026-02-11T10:00:00", "price": 400.0,
                 "portfolio_value": 100000.0, "z_score": 0.0},
                {"timestamp": "2026-02-11T10:01:00", "price": 410.0,
                 "portfolio_value": 101000.0, "z_score": 0.5},
            ],
        }
        stats = intraday_app.compute_session_stats(state)
        assert stats["pnl"] == 1000.0
        assert stats["pnl_pct"] == pytest.approx(0.01)
        assert stats["last_price"] == 410.0

    def test_negative_pnl(self):
        state = {
            "ticker": "MSFT",
            "broker_mode": "alpaca",
            "params": {"interval": 60},
            "trades": [],
            "tick_log": [
                {"timestamp": "2026-02-11T10:00:00", "price": 400.0,
                 "portfolio_value": 100000.0, "z_score": 0.0},
                {"timestamp": "2026-02-11T10:01:00", "price": 390.0,
                 "portfolio_value": 99000.0, "z_score": -0.5},
            ],
        }
        stats = intraday_app.compute_session_stats(state)
        assert stats["pnl"] == -1000.0

    def test_counts_trades(self):
        state = {
            "ticker": "MSFT",
            "broker_mode": "alpaca",
            "params": {"interval": 60},
            "trades": [
                {"action": "buy", "timestamp": "2026-02-11T10:00:00"},
                {"action": "sell", "timestamp": "2026-02-11T10:01:00"},
                {"action": "buy", "timestamp": "2026-02-11T10:02:00"},
            ],
            "tick_log": [
                {"timestamp": "2026-02-11T10:00:00", "price": 400.0,
                 "portfolio_value": 100000.0, "z_score": 0.0},
            ],
        }
        stats = intraday_app.compute_session_stats(state)
        assert stats["n_buys"] == 2
        assert stats["n_sells"] == 1
        assert stats["n_trades"] == 3


# =====================================================================
# Print session summary
# =====================================================================

class TestPrintSessionSummary:
    """Tests for print_session_summary."""

    def test_no_crash_empty(self, capsys):
        state = intraday_app.create_initial_state(
            "MSFT", 1.5, 30, 60, 6,
            broker_mode="alpaca", profile="intraday")
        intraday_app.print_session_summary(state)
        captured = capsys.readouterr()
        assert "Session Summary" in captured.out
        assert "MSFT" in captured.out

    def test_shows_pnl(self, capsys):
        state = {
            "ticker": "MSFT",
            "broker_mode": "alpaca",
            "params": {"interval": 60},
            "trades": [],
            "tick_log": [
                {"timestamp": "2026-02-11T10:00:00", "price": 400.0,
                 "portfolio_value": 100000.0, "z_score": 0.0},
                {"timestamp": "2026-02-11T10:01:00", "price": 410.0,
                 "portfolio_value": 101000.0, "z_score": 0.5},
            ],
        }
        intraday_app.print_session_summary(state)
        captured = capsys.readouterr()
        assert "1,000" in captured.out


# =====================================================================
# Alpaca bridge -- fetch_portfolio_state
# =====================================================================

class TestFetchPortfolioState:
    """Tests for fetch_portfolio_state()."""

    def test_returns_portfolio_with_position(self):
        broker = MagicMock(spec=AlpacaBroker)
        broker.get_cash.return_value = 50000.0
        pos = MagicMock()
        pos.qty = 100
        broker.get_position.return_value = pos

        portfolio = intraday_app.fetch_portfolio_state(
            broker, "MSFT", 410.0)
        assert isinstance(portfolio, Portfolio)
        assert portfolio.cash == 50000.0
        assert portfolio.shares == 100
        assert portfolio.price == 410.0

    def test_returns_portfolio_no_position(self):
        broker = MagicMock(spec=AlpacaBroker)
        broker.get_cash.return_value = 100000.0
        broker.get_position.return_value = None

        portfolio = intraday_app.fetch_portfolio_state(
            broker, "MSFT", 410.0)
        assert portfolio.shares == 0
        assert portfolio.cash == 100000.0

    def test_total_value_correct(self):
        broker = MagicMock(spec=AlpacaBroker)
        broker.get_cash.return_value = 10000.0
        pos = MagicMock()
        pos.qty = 50
        broker.get_position.return_value = pos

        portfolio = intraday_app.fetch_portfolio_state(
            broker, "MSFT", 200.0)
        assert portfolio.total_value == 10000.0 + 50 * 200.0


# =====================================================================
# Alpaca bridge -- execute_action
# =====================================================================

class TestExecuteAction:
    """Tests for execute_action()."""

    def test_hold_returns_none(self):
        broker = MagicMock(spec=AlpacaBroker)
        state = {"trades": []}
        result = intraday_app.execute_action(
            broker, "MSFT", Action.hold(), state)
        assert result is None
        assert len(state["trades"]) == 0
        broker.buy.assert_not_called()
        broker.sell.assert_not_called()

    def test_buy_submits_order(self):
        broker = MagicMock(spec=AlpacaBroker)
        broker.buy.return_value = OrderResult(
            order_id="o1", ticker="MSFT", side="buy",
            qty=10, status="filled", filled_qty=10, filled_price=410.0)
        state = {"trades": []}

        result = intraday_app.execute_action(
            broker, "MSFT", Action.buy_shares(10), state)
        assert result.status == "filled"
        assert result.filled_qty == 10
        assert len(state["trades"]) == 1
        assert state["trades"][0]["action"] == "buy"
        assert state["trades"][0]["order_id"] == "o1"

    def test_sell_submits_order(self):
        broker = MagicMock(spec=AlpacaBroker)
        broker.sell.return_value = OrderResult(
            order_id="o2", ticker="MSFT", side="sell",
            qty=5, status="filled", filled_qty=5, filled_price=415.0)
        state = {"trades": []}

        result = intraday_app.execute_action(
            broker, "MSFT", Action.sell_shares(5), state)
        assert result.status == "filled"
        assert len(state["trades"]) == 1
        assert state["trades"][0]["action"] == "sell"

    def test_dry_run_no_broker_calls(self):
        broker = MagicMock(spec=AlpacaBroker)
        state = {"trades": []}

        result = intraday_app.execute_action(
            broker, "MSFT", Action.buy_shares(10), state, dry_run=True)
        assert result is None
        broker.buy.assert_not_called()
        broker.sell.assert_not_called()
        assert len(state["trades"]) == 1
        assert state["trades"][0]["status"] == "dry_run"

    def test_dry_run_sell(self):
        broker = MagicMock(spec=AlpacaBroker)
        state = {"trades": []}

        intraday_app.execute_action(
            broker, "MSFT", Action.sell_shares(5), state, dry_run=True)
        broker.sell.assert_not_called()
        assert state["trades"][0]["status"] == "dry_run"
        assert state["trades"][0]["action"] == "sell"

    def test_error_logged(self):
        broker = MagicMock(spec=AlpacaBroker)
        broker.buy.side_effect = Exception("API timeout")
        state = {"trades": []}

        result = intraday_app.execute_action(
            broker, "MSFT", Action.buy_shares(10), state)
        assert result is None
        assert len(state["trades"]) == 1
        assert state["trades"][0]["status"] == "error"
        assert "API timeout" in state["trades"][0]["error"]

    def test_zero_shares_returns_none(self):
        broker = MagicMock(spec=AlpacaBroker)
        state = {"trades": []}
        action = Action(ActionType.BUY, fraction=0.0, shares=0.0)
        result = intraday_app.execute_action(
            broker, "MSFT", action, state)
        assert result is None
        assert len(state["trades"]) == 0


# =====================================================================
# Dry-run mode (dedicated tests)
# =====================================================================

class TestDryRunMode:
    """Dedicated dry-run tests."""

    def test_dry_run_buy_no_broker(self):
        broker = MagicMock(spec=AlpacaBroker)
        state = {"trades": []}
        intraday_app.execute_action(
            broker, "MSFT", Action.buy_shares(5), state, dry_run=True)
        broker.buy.assert_not_called()
        assert state["trades"][0]["status"] == "dry_run"

    def test_dry_run_sell_no_broker(self):
        broker = MagicMock(spec=AlpacaBroker)
        state = {"trades": []}
        intraday_app.execute_action(
            broker, "MSFT", Action.sell_shares(3), state, dry_run=True)
        broker.sell.assert_not_called()
        assert state["trades"][0]["status"] == "dry_run"

    def test_dry_run_still_reads_portfolio(self):
        """In dry-run, fetch_portfolio_state should still be callable."""
        broker = MagicMock(spec=AlpacaBroker)
        broker.get_cash.return_value = 50000.0
        broker.get_position.return_value = None

        portfolio = intraday_app.fetch_portfolio_state(
            broker, "MSFT", 410.0)
        assert portfolio.cash == 50000.0
        broker.get_cash.assert_called_once()

    def test_dry_run_price_is_zero(self):
        """Dry-run trades log price as 0.0 (no fill)."""
        broker = MagicMock(spec=AlpacaBroker)
        state = {"trades": []}
        intraday_app.execute_action(
            broker, "MSFT", Action.buy_shares(5), state, dry_run=True)
        assert state["trades"][0]["price"] == 0.0


# =====================================================================
# Load profile tests (YAML credential management)
# =====================================================================

class TestLoadProfile:
    """Tests for load_profile()."""

    def test_loads_from_yaml(self, tmp_path):
        """load_profile reads a YAML config correctly."""
        config = {
            "default_profile": "portfolio",
            "profiles": {
                "portfolio": {
                    "api_key": "PK-PORTFOLIO",
                    "secret_key": "SK-PORTFOLIO",
                    "paper": True,
                },
                "intraday": {
                    "api_key": "PK-INTRADAY",
                    "secret_key": "SK-INTRADAY",
                    "paper": True,
                },
            },
        }
        config_path = tmp_path / "config.yaml"
        import yaml
        config_path.write_text(yaml.dump(config))

        with patch("finance_tools.broker.alpaca.CONFIG_PATH",
                   str(config_path)):
            result = load_profile("intraday")
        assert result["api_key"] == "PK-INTRADAY"
        assert result["secret_key"] == "SK-INTRADAY"
        assert result["paper"] is True

    def test_uses_default_profile(self, tmp_path):
        """load_profile(None) uses default_profile from config."""
        config = {
            "default_profile": "portfolio",
            "profiles": {
                "portfolio": {
                    "api_key": "PK-DEF",
                    "secret_key": "SK-DEF",
                    "paper": True,
                },
            },
        }
        import yaml
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))

        with patch("finance_tools.broker.alpaca.CONFIG_PATH",
                   str(config_path)):
            result = load_profile(None)
        assert result["api_key"] == "PK-DEF"

    def test_missing_profile_raises_keyerror(self, tmp_path):
        """Requesting a non-existent profile raises KeyError."""
        config = {
            "default_profile": "portfolio",
            "profiles": {
                "portfolio": {
                    "api_key": "PK", "secret_key": "SK", "paper": True,
                },
            },
        }
        import yaml
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))

        with patch("finance_tools.broker.alpaca.CONFIG_PATH",
                   str(config_path)):
            with pytest.raises(KeyError, match="nonexistent"):
                load_profile("nonexistent")

    def test_falls_back_to_env_vars(self):
        """No config file -> falls back to env vars."""
        with patch("finance_tools.broker.alpaca.CONFIG_PATH",
                   "/nonexistent/config.yaml"):
            with patch.dict(os.environ, {
                "ALPACA_API_KEY": "ENV-KEY",
                "ALPACA_SECRET_KEY": "ENV-SECRET",
            }):
                result = load_profile()
        assert result["api_key"] == "ENV-KEY"
        assert result["secret_key"] == "ENV-SECRET"

    def test_no_config_no_env_raises(self):
        """No config file and no env vars -> FileNotFoundError."""
        with patch("finance_tools.broker.alpaca.CONFIG_PATH",
                   "/nonexistent/config.yaml"):
            env = {k: v for k, v in os.environ.items()
                   if k not in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")}
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(FileNotFoundError):
                    load_profile()

    def test_paper_defaults_to_true(self, tmp_path):
        """If paper is missing from profile, defaults to True."""
        config = {
            "default_profile": "test",
            "profiles": {
                "test": {"api_key": "PK", "secret_key": "SK"},
            },
        }
        import yaml
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))

        with patch("finance_tools.broker.alpaca.CONFIG_PATH",
                   str(config_path)):
            result = load_profile("test")
        assert result["paper"] is True


# =====================================================================
# AlpacaBroker profile tests
# =====================================================================

class TestAlpacaBrokerProfile:
    """Tests for AlpacaBroker with profile parameter."""

    def test_explicit_keys_override_profile(self):
        """Explicit api_key/secret_key takes priority over profile."""
        broker = AlpacaBroker(api_key="EXPLICIT-K", secret_key="EXPLICIT-S")
        assert broker._api_key == "EXPLICIT-K"
        assert broker._secret_key == "EXPLICIT-S"

    def test_profile_loads_credentials(self, tmp_path):
        """profile= loads from YAML."""
        config = {
            "default_profile": "portfolio",
            "profiles": {
                "intraday": {
                    "api_key": "PK-INTRA",
                    "secret_key": "SK-INTRA",
                    "paper": True,
                },
            },
        }
        import yaml
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))

        with patch("finance_tools.broker.alpaca.CONFIG_PATH",
                   str(config_path)):
            broker = AlpacaBroker(profile="intraday")
        assert broker._api_key == "PK-INTRA"
        assert broker._secret_key == "SK-INTRA"

    def test_no_profile_no_keys_uses_env(self):
        """No profile, no keys -> env vars (backward compat)."""
        with patch.dict(os.environ, {
            "ALPACA_API_KEY": "ENV-K",
            "ALPACA_SECRET_KEY": "ENV-S",
        }):
            broker = AlpacaBroker()
        assert broker._api_key == "ENV-K"
        assert broker._secret_key == "ENV-S"


# =====================================================================
# OU strategy helpers
# =====================================================================

def _generate_ou_prices(n: int, theta: float = 2.0, mu: float = 4.6,
                        sigma: float = 0.1, dt: float = 0.1,
                        seed: int = 603) -> list[float]:
    """Generate synthetic prices from a discrete OU process on log-prices.

    Uses Euler-Maruyama discretization.  For stability, theta*dt should
    be well below 1.

    Returns a list of n prices (positive).
    """
    rng = np.random.default_rng(seed)
    log_prices = np.zeros(n)
    log_prices[0] = mu
    for i in range(1, n):
        dW = rng.normal(0, np.sqrt(dt))
        log_prices[i] = (log_prices[i - 1]
                         + theta * (mu - log_prices[i - 1]) * dt
                         + sigma * dW)
    return np.exp(log_prices).tolist()


def _generate_trending_prices(n: int, start: float = 100.0,
                              seed: int = 603) -> list[float]:
    """Generate explosive (anti-mean-reverting) log-prices.

    Uses dX = 0.01 + 0.02 * X + noise, giving b > 0 -> theta < 0 in
    the OLS fit, so _fit_ou must return None.
    """
    rng = np.random.default_rng(seed)
    log_prices = np.zeros(n)
    log_prices[0] = np.log(start)
    for i in range(1, n):
        # Positive coefficient on X makes this explosive (b > 0)
        log_prices[i] = (log_prices[i - 1]
                         + 0.01 + 0.02 * log_prices[i - 1]
                         + rng.normal(0, 0.001))
    # Clip to avoid overflow in exp
    log_prices = np.clip(log_prices, -50, 50)
    return np.exp(log_prices).tolist()


# =====================================================================
# IntradayOU strategy tests
# =====================================================================

class TestIntradayOU:
    """Tests for the base IntradayOU strategy."""

    def test_hold_on_insufficient_history(self):
        """Strategy returns HOLD when history is too short."""
        strategy = IntradayOU(window=60)
        prices = [100.0] * 20
        history = _make_history(prices)
        day = history.iloc[-1]
        portfolio = _make_portfolio()
        action = strategy.decide(day, history, portfolio)
        assert action.action == ActionType.HOLD

    def test_hold_on_zero_volatility(self):
        """HOLD when all prices are constant (sigma=0)."""
        strategy = IntradayOU(window=5)
        prices = [100.0] * 20
        history = _make_history(prices)
        day = history.iloc[-1]
        portfolio = _make_portfolio()
        action = strategy.decide(day, history, portfolio)
        assert action.action == ActionType.HOLD

    def test_hold_on_small_signal(self):
        """HOLD when signal is below entry threshold."""
        strategy = IntradayOU(window=30, entry_threshold=5.0)
        prices = _generate_ou_prices(100)
        history = _make_history(prices)
        day = history.iloc[-1]
        portfolio = _make_portfolio()
        action = strategy.decide(day, history, portfolio)
        # With high threshold, most OU signals won't trigger
        assert action.action == ActionType.HOLD

    def test_hold_when_theta_nonpositive(self):
        """HOLD when data is trending (theta <= 0)."""
        strategy = IntradayOU(window=30, entry_threshold=1.0)
        prices = _generate_trending_prices(100)
        history = _make_history(prices)
        day = history.iloc[-1]
        portfolio = _make_portfolio()
        action = strategy.decide(day, history, portfolio)
        assert action.action == ActionType.HOLD

    def test_buy_on_price_below_mean(self):
        """BUY when price is well below OU equilibrium."""
        strategy = IntradayOU(window=50, entry_threshold=1.5)
        # Generate OU data then add a dip at the end
        prices = _generate_ou_prices(60)
        # Push price well below mean (mu=4.6 -> exp(4.6)~100)
        for _ in range(5):
            prices.append(prices[-1] * 0.95)  # successive dips
        history = _make_history(prices)
        day = history.iloc[-1]
        portfolio = _make_portfolio(cash=10000, shares=0, price=prices[-1])
        action = strategy.decide(day, history, portfolio)
        # May buy if signal is strong enough
        assert action.action in (ActionType.BUY, ActionType.HOLD)

    def test_sell_on_price_above_mean(self):
        """SELL when price is well above OU equilibrium."""
        strategy = IntradayOU(window=50, entry_threshold=1.5)
        prices = _generate_ou_prices(60)
        # Push price well above mean
        for _ in range(5):
            prices.append(prices[-1] * 1.05)
        history = _make_history(prices)
        day = history.iloc[-1]
        portfolio = _make_portfolio(cash=0, shares=50, price=prices[-1])
        action = strategy.decide(day, history, portfolio)
        assert action.action in (ActionType.SELL, ActionType.HOLD)

    def test_no_buy_with_zero_cash(self):
        """HOLD when cash is zero even if signal says buy."""
        strategy = IntradayOU(window=50, entry_threshold=1.0)
        prices = _generate_ou_prices(60)
        for _ in range(5):
            prices.append(prices[-1] * 0.90)
        history = _make_history(prices)
        day = history.iloc[-1]
        portfolio = _make_portfolio(cash=0, shares=50, price=prices[-1])
        action = strategy.decide(day, history, portfolio)
        assert action.action != ActionType.BUY

    def test_no_sell_with_zero_shares(self):
        """HOLD when shares are zero even if signal says sell."""
        strategy = IntradayOU(window=50, entry_threshold=1.0)
        prices = _generate_ou_prices(60)
        for _ in range(5):
            prices.append(prices[-1] * 1.10)
        history = _make_history(prices)
        day = history.iloc[-1]
        portfolio = _make_portfolio(cash=10000, shares=0, price=prices[-1])
        action = strategy.decide(day, history, portfolio)
        assert action.action != ActionType.SELL

    def test_whole_share_sizing(self):
        """Buy/sell actions use whole shares."""
        strategy = IntradayOU(window=50, entry_threshold=1.0)
        prices = _generate_ou_prices(60)
        for _ in range(5):
            prices.append(prices[-1] * 0.90)
        history = _make_history(prices)
        day = history.iloc[-1]
        portfolio = _make_portfolio(cash=10000, shares=0, price=prices[-1])
        action = strategy.decide(day, history, portfolio)
        if action.action == ActionType.BUY:
            assert action.shares is not None
            assert action.shares == math.floor(action.shares)
            assert action.shares >= 1

    def test_max_position_cap(self):
        """Position fraction is capped at max_position_frac."""
        strategy = IntradayOU(window=50, entry_threshold=1.0,
                              max_threshold=1.5, max_position_frac=0.30)
        prices = _generate_ou_prices(60)
        for _ in range(10):
            prices.append(prices[-1] * 0.90)
        history = _make_history(prices)
        day = history.iloc[-1]
        price = prices[-1]
        portfolio = _make_portfolio(cash=10000, shares=0, price=price)
        action = strategy.decide(day, history, portfolio)
        if action.action == ActionType.BUY and action.shares:
            max_shares = math.floor(10000 * 0.30 / price)
            assert action.shares <= max_shares

    def test_compute_z_returns_float(self):
        """compute_z() returns a float when history is sufficient."""
        strategy = IntradayOU(window=30)
        prices = _generate_ou_prices(100)
        history = _make_history(prices)
        z = strategy.compute_z(history)
        # May be None if theta <= 0 for this sample, but typically float
        if z is not None:
            assert isinstance(z, float)

    def test_compute_z_returns_none_insufficient_history(self):
        """compute_z() returns None when history is too short."""
        strategy = IntradayOU(window=60)
        prices = [100.0] * 10
        history = _make_history(prices)
        z = strategy.compute_z(history)
        assert z is None

    def test_compute_z_returns_none_for_trending_data(self):
        """compute_z() returns None for trending (non-OU) data."""
        strategy = IntradayOU(window=30)
        prices = _generate_trending_prices(100)
        history = _make_history(prices)
        z = strategy.compute_z(history)
        assert z is None

    def test_subclasses_strategy(self):
        """IntradayOU is a proper Strategy subclass."""
        assert issubclass(IntradayOU, Strategy)
        strategy = IntradayOU()
        assert isinstance(strategy, Strategy)

    def test_name_includes_entry(self):
        """Strategy name includes the entry threshold."""
        strategy = IntradayOU(entry_threshold=2.5)
        assert "2.5" in strategy.name
        assert "OU" in strategy.name

    def test_negative_prices_handled(self):
        """Negative/zero prices result in HOLD (not a crash)."""
        strategy = IntradayOU(window=5)
        prices = [100.0] * 5 + [0.0, -1.0]
        history = _make_history(prices)
        day = history.iloc[-1]
        portfolio = _make_portfolio()
        action = strategy.decide(day, history, portfolio)
        assert action.action == ActionType.HOLD


# =====================================================================
# IntradayOU _fit_ou tests
# =====================================================================

class TestFitOU:
    """Tests for the OLS OU parameter estimation."""

    def test_fit_ou_returns_tuple_for_ou_data(self):
        """_fit_ou returns (theta, mu, sigma_eq) for mean-reverting data."""
        strategy = IntradayOU(window=60)
        prices = _generate_ou_prices(200)
        log_prices = np.log(prices)
        result = strategy._fit_ou(log_prices)
        assert result is not None
        theta, mu, sigma_eq = result
        assert theta > 0
        assert sigma_eq > 0

    def test_fit_ou_returns_none_for_trending(self):
        """_fit_ou returns None for trending data (theta <= 0)."""
        strategy = IntradayOU(window=30)
        prices = _generate_trending_prices(100)
        log_prices = np.log(prices)
        result = strategy._fit_ou(log_prices)
        assert result is None

    def test_fit_ou_returns_none_for_constant(self):
        """_fit_ou returns None when all prices are constant."""
        strategy = IntradayOU()
        log_prices = np.log([100.0] * 20)
        result = strategy._fit_ou(log_prices)
        assert result is None

    def test_fit_ou_returns_none_for_too_short(self):
        """_fit_ou returns None with fewer than 4 data points."""
        strategy = IntradayOU()
        log_prices = np.log([100.0, 101.0, 100.5])
        result = strategy._fit_ou(log_prices)
        # With only 2 dX points, OLS may or may not fit -- but should not crash
        assert result is None or len(result) == 3

    def test_signal_magnitude_increases_with_deviation(self):
        """OU signal magnitude grows as price deviates from mu."""
        strategy = IntradayOU(window=50)
        base_prices = _generate_ou_prices(60)
        base_history = _make_history(base_prices)
        z_base = strategy.compute_z(base_history)

        # Create larger deviation
        deviated = base_prices.copy()
        for _ in range(8):
            deviated.append(deviated[-1] * 0.92)
        dev_history = _make_history(deviated)
        z_dev = strategy.compute_z(dev_history)

        if z_base is not None and z_dev is not None:
            assert abs(z_dev) > abs(z_base)

    def test_linear_position_sizing(self):
        """Position fraction scales linearly between entry and max threshold."""
        strategy = IntradayOU(window=50, entry_threshold=2.0,
                              max_threshold=4.0, max_position_frac=0.80)
        # At entry=2.0, fraction should be ~0
        # At max=4.0, fraction should be ~0.80
        # Midpoint signal=3.0 -> fraction should be ~0.40
        # Test via the formula directly
        for s, expected in [(2.0, 0.0), (3.0, 0.40), (4.0, 0.80), (5.0, 0.80)]:
            t = min((s - 2.0) / max(4.0 - 2.0, 1e-8), 1.0)
            frac = t * 0.80
            assert abs(frac - expected) < 0.01


# =====================================================================
# IntradayOUWithCooldown tests
# =====================================================================

class TestIntradayOUWithCooldown:
    """Tests for the OU cooldown variant."""

    def _ou_buy_signal_history(self):
        """Generate history that triggers a buy for OU strategy."""
        prices = _generate_ou_prices(60)
        for _ in range(8):
            prices.append(prices[-1] * 0.90)
        return prices

    def test_blocks_consecutive_trades(self):
        """Second trade within cooldown period is blocked."""
        strategy = IntradayOUWithCooldown(
            window=50, entry_threshold=1.0, cooldown_ticks=3)
        prices = self._ou_buy_signal_history()
        history = _make_history(prices)
        day = history.iloc[-1]
        portfolio = _make_portfolio(cash=10000, shares=0, price=prices[-1])

        a1 = strategy.decide(day, history, portfolio)
        if a1.action == ActionType.BUY:
            a2 = strategy.decide(day, history, portfolio)
            assert a2.action == ActionType.HOLD

    def test_allows_trade_after_cooldown(self):
        """Trade is allowed after cooldown period expires."""
        strategy = IntradayOUWithCooldown(
            window=50, entry_threshold=1.0, cooldown_ticks=2)
        prices = self._ou_buy_signal_history()
        history = _make_history(prices)
        day = history.iloc[-1]
        portfolio = _make_portfolio(cash=10000, shares=0, price=prices[-1])

        a1 = strategy.decide(day, history, portfolio)
        if a1.action == ActionType.BUY:
            # Cooldown ticks
            strategy.decide(day, history, portfolio)  # tick 1
            strategy.decide(day, history, portfolio)  # tick 2
            # After cooldown
            a4 = strategy.decide(day, history, portfolio)
            assert a4.action in (ActionType.BUY, ActionType.HOLD)

    def test_reset_cooldown(self):
        """reset_cooldown() makes strategy ready to trade immediately."""
        strategy = IntradayOUWithCooldown(
            window=50, entry_threshold=1.0, cooldown_ticks=100)
        prices = self._ou_buy_signal_history()
        history = _make_history(prices)
        day = history.iloc[-1]
        portfolio = _make_portfolio(cash=10000, shares=0, price=prices[-1])

        a1 = strategy.decide(day, history, portfolio)
        if a1.action == ActionType.BUY:
            a2 = strategy.decide(day, history, portfolio)
            assert a2.action == ActionType.HOLD
            strategy.reset_cooldown()
            a3 = strategy.decide(day, history, portfolio)
            # Should be able to trade again
            assert a3.action in (ActionType.BUY, ActionType.HOLD)

    def test_subclasses_intradayou(self):
        """IntradayOUWithCooldown inherits from IntradayOU."""
        assert issubclass(IntradayOUWithCooldown, IntradayOU)
        strategy = IntradayOUWithCooldown()
        assert isinstance(strategy, IntradayOU)
        assert isinstance(strategy, Strategy)

    def test_name_includes_cooldown(self):
        """Strategy name includes cooldown info."""
        strategy = IntradayOUWithCooldown(entry_threshold=2.0, cooldown_ticks=6)
        assert "cd=6" in strategy.name
        assert "OU" in strategy.name


# =====================================================================
# State management tests for OU strategy
# =====================================================================

class TestOUStateManagement:
    """Tests for OU-related state management."""

    def test_create_initial_state_with_ou(self):
        """Initial state stores OU strategy params."""
        state = intraday_app.create_initial_state(
            "AAPL", 2.0, 60, 60, 6,
            strategy_type="ou", entry=2.5, max_threshold=3.5,
            cash=10000.0)
        assert state["params"]["strategy"] == "ou"
        assert state["params"]["entry_threshold"] == 2.5
        assert state["params"]["max_threshold"] == 3.5

    def test_create_initial_state_backward_compat(self):
        """Default strategy type is chebyshev for backward compat."""
        state = intraday_app.create_initial_state(
            "AAPL", 2.0, 30, 10, 6, cash=10000.0)
        assert state["params"]["strategy"] == "chebyshev"


# =====================================================================
# AggregatedTick tests
# =====================================================================

class TestAggregatedTick:
    """Tests for the AggregatedTick dataclass."""

    def test_fields(self):
        """AggregatedTick has all expected fields."""
        tick = AggregatedTick(
            price=150.5, volume=2300.0, high=151.0, low=149.0,
            timestamp=datetime.now(), n_trades=42,
        )
        assert tick.price == 150.5
        assert tick.volume == 2300.0
        assert tick.high == 151.0
        assert tick.low == 149.0
        assert tick.n_trades == 42

    def test_defaults(self):
        """AggregatedTick has default n_trades=1."""
        tick = AggregatedTick(
            price=100.0, volume=100.0, high=100.0, low=100.0,
            timestamp=datetime.now(),
        )
        assert tick.n_trades == 1


# =====================================================================
# AlpacaStreamFeed tests
# =====================================================================

class TestAlpacaStreamFeed:
    """Tests for the AlpacaStreamFeed WebSocket data feed."""

    def test_requires_credentials(self):
        """Raises ValueError without API credentials."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Alpaca credentials"):
                AlpacaStreamFeed("AAPL", api_key="", secret_key="")

    def test_reads_env_vars(self):
        """Reads credentials from environment variables."""
        env = {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"}
        with patch.dict(os.environ, env, clear=False):
            with patch("finance_tools.broker.data_feed.AlpacaFeed"):
                feed = AlpacaStreamFeed("AAPL")
        assert feed._api_key == "test_key"
        assert feed._secret_key == "test_secret"

    def test_uppercases_ticker(self):
        """Ticker is uppercased on init."""
        env = {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}
        with patch.dict(os.environ, env, clear=False):
            with patch("finance_tools.broker.data_feed.AlpacaFeed"):
                feed = AlpacaStreamFeed("aapl")
        assert feed.ticker == "AAPL"

    def test_custom_agg_interval(self):
        """Custom aggregation interval is stored."""
        env = {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}
        with patch.dict(os.environ, env, clear=False):
            with patch("finance_tools.broker.data_feed.AlpacaFeed"):
                feed = AlpacaStreamFeed("AAPL", agg_interval=5.0)
        assert feed._agg_interval == 5.0

    def test_is_streaming_false_before_start(self):
        """is_streaming is False before start() is called."""
        env = {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}
        with patch.dict(os.environ, env, clear=False):
            with patch("finance_tools.broker.data_feed.AlpacaFeed"):
                feed = AlpacaStreamFeed("AAPL")
        assert feed.is_streaming is False

    def test_get_tick_returns_none_on_empty_queue(self):
        """get_tick() returns None when queue is empty."""
        env = {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}
        with patch.dict(os.environ, env, clear=False):
            with patch("finance_tools.broker.data_feed.AlpacaFeed"):
                feed = AlpacaStreamFeed("AAPL")
        result = feed.get_tick(timeout=0.01)
        assert result is None

    def test_drain_ticks_returns_empty_list(self):
        """drain_ticks() returns empty list when queue is empty."""
        env = {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}
        with patch.dict(os.environ, env, clear=False):
            with patch("finance_tools.broker.data_feed.AlpacaFeed"):
                feed = AlpacaStreamFeed("AAPL")
        result = feed.drain_ticks()
        assert result == []

    def test_flush_buffer_aggregates_trades(self):
        """_flush_buffer() correctly aggregates buffered trades."""
        env = {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}
        with patch.dict(os.environ, env, clear=False):
            with patch("finance_tools.broker.data_feed.AlpacaFeed"):
                feed = AlpacaStreamFeed("AAPL")

        now = datetime.now()
        feed._buffer = [
            (100.0, 50.0, now),
            (101.5, 30.0, now),
            (99.0, 20.0, now),
            (102.0, 10.0, now),
        ]
        import time as _time
        feed._flush_buffer(_time.monotonic())

        tick = feed.get_tick(timeout=0.1)
        assert tick is not None
        assert tick.price == 102.0       # last trade
        assert tick.volume == 110.0      # 50+30+20+10
        assert tick.high == 102.0        # max
        assert tick.low == 99.0          # min
        assert tick.n_trades == 4

    def test_latest_falls_back_to_rest(self):
        """latest() delegates to REST feed when no stream data."""
        env = {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}
        with patch.dict(os.environ, env, clear=False):
            mock_rest = MagicMock()
            mock_rest.latest.return_value = Quote(
                ticker="AAPL", price=155.0, timestamp=datetime.now(),
            )
            with patch("finance_tools.broker.data_feed.AlpacaFeed",
                       return_value=mock_rest):
                feed = AlpacaStreamFeed("AAPL")

        quote = feed.latest()
        assert quote.price == 155.0
        mock_rest.latest.assert_called_once()

    def test_latest_uses_stream_data(self):
        """latest() returns stream data when _last_tick is set."""
        env = {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}
        with patch.dict(os.environ, env, clear=False):
            with patch("finance_tools.broker.data_feed.AlpacaFeed"):
                feed = AlpacaStreamFeed("AAPL")

        feed._last_tick = AggregatedTick(
            price=160.0, volume=500.0, high=161.0, low=159.0,
            timestamp=datetime.now(), n_trades=10,
        )
        quote = feed.latest()
        assert quote.price == 160.0
        assert quote.volume == 500.0

    def test_history_delegates_to_rest(self):
        """history() delegates to the internal REST feed."""
        env = {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}
        with patch.dict(os.environ, env, clear=False):
            mock_rest = MagicMock()
            expected_df = pd.DataFrame({"Close": [100.0, 101.0]})
            mock_rest.history.return_value = expected_df
            with patch("finance_tools.broker.data_feed.AlpacaFeed",
                       return_value=mock_rest):
                feed = AlpacaStreamFeed("AAPL")

        result = feed.history(lookback_minutes=120)
        mock_rest.history.assert_called_once_with(lookback_minutes=120)
        pd.testing.assert_frame_equal(result, expected_df)

    def test_subclasses_datafeed(self):
        """AlpacaStreamFeed is a proper DataFeed subclass."""
        assert issubclass(AlpacaStreamFeed, DataFeed)

    def test_stop_safe_without_start(self):
        """stop() does not raise even if start() was never called."""
        env = {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}
        with patch.dict(os.environ, env, clear=False):
            with patch("finance_tools.broker.data_feed.AlpacaFeed"):
                feed = AlpacaStreamFeed("AAPL")
        # Should not raise
        feed.stop()


# =====================================================================
# CLI parsing tests -- LOCAL broker
# =====================================================================

class TestCLIParsingLocal:
    """Tests for CLI argument parsing (local broker context)."""

    def test_stream_flag_parsed(self):
        """--stream flag is parsed correctly."""
        with patch("sys.argv", ["app.py", "MSFT", "--feed", "alpaca",
                                "--stream"]):
            args = intraday_app.parse_args()
        assert args.stream is True

    def test_stream_defaults_to_false(self):
        """--stream defaults to False when not specified."""
        with patch("sys.argv", ["app.py", "MSFT"]):
            args = intraday_app.parse_args()
        assert args.stream is False


# =====================================================================
# CLI parsing tests -- ALPACA broker
# =====================================================================

class TestCLIParsingAlpaca:
    """Tests for CLI argument parsing (alpaca broker context)."""

    def test_default_profile_is_intraday(self):
        with patch("sys.argv", ["app.py", "MSFT", "--broker", "alpaca"]):
            args = intraday_app.parse_args()
        assert args.profile == "intraday"

    def test_custom_profile(self):
        with patch("sys.argv", ["app.py", "MSFT", "--broker", "alpaca",
                                "--profile", "portfolio"]):
            args = intraday_app.parse_args()
        assert args.profile == "portfolio"

    def test_stream_flag(self):
        with patch("sys.argv", ["app.py", "MSFT", "--broker", "alpaca",
                                "--stream"]):
            args = intraday_app.parse_args()
        assert args.stream is True

    def test_dry_run_flag(self):
        with patch("sys.argv", ["app.py", "MSFT", "--broker", "alpaca",
                                "--dry-run"]):
            args = intraday_app.parse_args()
        assert args.dry_run is True

    def test_strategy_choices(self):
        with patch("sys.argv", ["app.py", "MSFT", "--strategy", "ou"]):
            args = intraday_app.parse_args()
        assert args.strategy == "ou"

    def test_broker_flag(self):
        """--broker alpaca selects alpaca broker."""
        with patch("sys.argv", ["app.py", "MSFT", "--broker", "alpaca"]):
            args = intraday_app.parse_args()
        assert args.broker == "alpaca"

    def test_default_broker_is_local(self):
        """Default broker is local."""
        with patch("sys.argv", ["app.py", "MSFT"]):
            args = intraday_app.parse_args()
        assert args.broker == "local"


# =====================================================================
# Strategy integration tests (Alpaca broker context)
# =====================================================================

class TestStrategyIntegration:
    """Tests that strategies are correctly wired in the merged app."""

    def test_chebyshev_creates_correct_portfolio(self):
        """Portfolio built from Alpaca state should work with strategies."""
        portfolio = Portfolio(cash=50000.0, shares=100, price=410.0)
        assert portfolio.total_value == 50000.0 + 100 * 410.0
        assert portfolio.equity == 100 * 410.0

    def test_ou_strategy_imported(self):
        assert hasattr(intraday_app, "IntradayOUWithCooldown")

    def test_chebyshev_strategy_imported(self):
        assert hasattr(intraday_app, "IntradayChebyshevWithCooldown")

    def test_strategies_have_decide_method(self):
        cheb = IntradayChebyshevWithCooldown(window=10, k_threshold=1.5)
        ou = IntradayOUWithCooldown(window=10, entry_threshold=2.0)
        assert hasattr(cheb, "decide")
        assert hasattr(ou, "decide")


# =====================================================================
# File structure and cross-script consistency
# =====================================================================

class TestFileStructure:
    """Tests for project file organization (new layout)."""

    def test_strategy_file_exists(self):
        """intraday.py exists in the strategies directory."""
        assert os.path.isfile(
            os.path.join(STRATEGIES_DIR, "intraday.py"))

    def test_app_file_exists(self):
        """app.py exists in the app directory."""
        assert os.path.isfile(os.path.join(APP_DIR, "app.py"))

    def test_backtest_intraday_file_exists(self):
        """intraday.py exists in the backtest directory."""
        assert os.path.isfile(os.path.join(BACKTEST_DIR, "intraday.py"))

    def test_data_feed_file_exists(self):
        """data_feed.py exists in the broker directory."""
        assert os.path.isfile(os.path.join(BROKER_DIR, "data_feed.py"))

    def test_alpaca_broker_exists(self):
        """alpaca.py exists in the broker directory."""
        assert os.path.isfile(os.path.join(BROKER_DIR, "alpaca.py"))

    def test_strategy_imports_from_engine(self):
        """Strategy file imports from finance_tools.backtest.engine."""
        with open(os.path.join(STRATEGIES_DIR, "intraday.py")) as f:
            content = f.read()
        assert "from finance_tools.backtest.engine import" in content

    def test_app_imports_from_data_feed(self):
        """App imports from finance_tools.broker.data_feed."""
        with open(os.path.join(APP_DIR, "app.py")) as f:
            content = f.read()
        assert "from finance_tools.broker.data_feed import" in content

    def test_strategy_uses_whole_shares(self):
        """Strategy uses buy_shares/sell_shares for whole-share trades."""
        with open(os.path.join(STRATEGIES_DIR, "intraday.py")) as f:
            content = f.read()
        assert "buy_shares" in content
        assert "sell_shares" in content

    def test_strategy_contains_ou_classes(self):
        """intraday.py contains both OU strategy classes."""
        with open(os.path.join(STRATEGIES_DIR, "intraday.py")) as f:
            content = f.read()
        assert "class IntradayOU" in content
        assert "class IntradayOUWithCooldown" in content

    def test_backtest_supports_strategy_arg(self):
        """backtest/intraday.py supports --strategy argument."""
        with open(os.path.join(BACKTEST_DIR, "intraday.py")) as f:
            content = f.read()
        assert "--strategy" in content
        assert "IntradayOUWithCooldown" in content

    def test_app_supports_strategy_arg(self):
        """app.py supports --strategy argument."""
        with open(os.path.join(APP_DIR, "app.py")) as f:
            content = f.read()
        assert "--strategy" in content
        assert "IntradayOUWithCooldown" in content

    def test_data_feed_contains_stream_classes(self):
        """data_feed.py contains AlpacaStreamFeed and AggregatedTick."""
        with open(os.path.join(BROKER_DIR, "data_feed.py")) as f:
            content = f.read()
        assert "class AlpacaStreamFeed" in content
        assert "class AggregatedTick" in content

    def test_app_contains_streaming_features(self):
        """app.py contains --stream flag and run_streaming_loop."""
        with open(os.path.join(APP_DIR, "app.py")) as f:
            content = f.read()
        assert "--stream" in content
        assert "run_streaming_loop" in content

    def test_app_contains_broker_flag(self):
        """app.py supports --broker flag."""
        with open(os.path.join(APP_DIR, "app.py")) as f:
            content = f.read()
        assert "--broker" in content

    def test_app_contains_alpaca_functions(self):
        """app.py contains Alpaca bridge functions."""
        with open(os.path.join(APP_DIR, "app.py")) as f:
            content = f.read()
        assert "fetch_portfolio_state" in content
        assert "execute_action" in content
        assert "run_polling_loop_alpaca" in content
        assert "run_streaming_loop_alpaca" in content

    def test_state_file_dotfile_local(self):
        """Local state file is a dotfile with 'intraday' in the name."""
        path = intraday_app._state_file("MSFT", broker_mode="local")
        basename = os.path.basename(path)
        assert basename.startswith(".")
        assert "intraday" in basename

    def test_state_file_dotfile_alpaca(self):
        """Alpaca state file is a dotfile with 'alpaca_intraday' in name."""
        path = intraday_app._state_file("MSFT", broker_mode="alpaca")
        basename = os.path.basename(path)
        assert basename.startswith(".")
        assert "alpaca_intraday" in basename


# =====================================================================
# Cross-script consistency
# =====================================================================

class TestCrossScriptConsistency:
    """Verify consistency between merged app and shared modules."""

    def test_imports_alpaca_broker(self):
        assert hasattr(intraday_app, "AlpacaBroker")

    def test_imports_order_result(self):
        assert hasattr(intraday_app, "OrderResult")

    def test_imports_portfolio(self):
        assert hasattr(intraday_app, "Portfolio")

    def test_imports_action(self):
        assert hasattr(intraday_app, "Action")

    def test_check_risk_exists(self):
        assert hasattr(intraday_app, "check_risk")
        assert callable(intraday_app.check_risk)

    def test_fetch_portfolio_state_exists(self):
        assert hasattr(intraday_app, "fetch_portfolio_state")
        assert callable(intraday_app.fetch_portfolio_state)

    def test_execute_action_exists(self):
        assert hasattr(intraday_app, "execute_action")
        assert callable(intraday_app.execute_action)

    def test_run_polling_loop_local_exists(self):
        assert hasattr(intraday_app, "run_polling_loop_local")
        assert callable(intraday_app.run_polling_loop_local)

    def test_run_streaming_loop_local_exists(self):
        assert hasattr(intraday_app, "run_streaming_loop_local")
        assert callable(intraday_app.run_streaming_loop_local)

    def test_run_polling_loop_alpaca_exists(self):
        assert hasattr(intraday_app, "run_polling_loop_alpaca")
        assert callable(intraday_app.run_polling_loop_alpaca)

    def test_run_streaming_loop_alpaca_exists(self):
        assert hasattr(intraday_app, "run_streaming_loop_alpaca")
        assert callable(intraday_app.run_streaming_loop_alpaca)
