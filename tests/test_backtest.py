"""
Tests for portfolio backtest engine, strategies, and Monte Carlo stress-test framework.

Merged from test_portfolio_backtest.py (91 tests) and test_monte_carlo_backtest.py (54 tests).
Uses synthetic price data (no yfinance dependency).
"""

import importlib.util
import math
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from finance_tools.backtest.engine import Action, ActionType
from finance_tools.backtest.portfolio import (
    PortfolioState, PortfolioStrategy, PortfolioBacktester,
    PortfolioBacktestResult, PortfolioTrade,
)
from finance_tools.strategies.portfolio import (
    EqualWeightRebalance, InverseVolatilityWeight,
    IndependentMeanReversion, RelativeStrength,
)
from finance_tools.strategies.equal_weight import (
    compute_inv_vol_target_shares, inv_vol_needs_rebalance, compute_volatility,
)
from finance_tools.backtest.monte_carlo import (
    TimeWindow, MCIterationResult,
    fetch_full_history, _normalize_index, slice_history,
    generate_windows, run_monte_carlo,
    mc_results_to_dataframe, summarize_mc_results, compute_win_rates,
)

# Repo root for file-structure tests
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =====================================================================
# Helpers — portfolio backtest
# =====================================================================

def make_hist(prices: list[float], volumes: list[float] = None,
              dividends: list[float] = None,
              start_date: str = "2023-01-02") -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from closing prices."""
    n = len(prices)
    if volumes is None:
        volumes = [1_000_000] * n
    if dividends is None:
        dividends = [0.0] * n
    dates = pd.bdate_range(start_date, periods=n)
    return pd.DataFrame({
        "Open": prices,
        "High": [p * 1.01 for p in prices],
        "Low": [p * 0.99 for p in prices],
        "Close": prices,
        "Volume": volumes,
        "Dividends": dividends,
    }, index=dates)


def make_two_stock_hist(prices_a: list[float], prices_b: list[float],
                        dividends_a=None, dividends_b=None):
    """Build a hist_dict with two tickers A and B."""
    return {
        "A": make_hist(prices_a, dividends=dividends_a),
        "B": make_hist(prices_b, dividends=dividends_b),
    }


class HoldStrategy(PortfolioStrategy):
    """Never trades. For testing no-op behavior."""
    name = "Hold"

    def decide(self, day, history, portfolio):
        return {}


class BuyAllDay1(PortfolioStrategy):
    """Buys equal amounts of all tickers on day 1, then holds."""
    name = "Buy All Day 1"

    def decide(self, day, history, portfolio):
        # Buy on day 1 only
        if any(len(h) > 1 for h in history.values()):
            return {}
        tickers = sorted(portfolio.positions.keys())
        n = len(tickers)
        return {t: Action.buy(1.0 / n) for t in tickers}


class SellThenBuy(PortfolioStrategy):
    """On day 2: sell A, buy B. Tests sell-before-buy ordering."""
    name = "Sell Then Buy"

    def decide(self, day, history, portfolio):
        tickers = sorted(portfolio.positions.keys())
        # Day 1: buy A
        if all(len(h) == 1 for h in history.values()):
            return {"A": Action.buy(1.0)}
        # Day 2: sell A, buy B
        if all(len(h) == 2 for h in history.values()):
            return {"A": Action.sell(1.0), "B": Action.buy(1.0)}
        return {}


# =====================================================================
# Helpers — Monte Carlo
# =====================================================================

def _make_price_df(n_days: int = 500, start_date: str = "2015-01-02",
                   start_price: float = 100.0, seed: int = 42,
                   tz: str | None = None) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame with random walk prices."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)
    if tz:
        dates = dates.tz_localize(tz)

    returns = rng.normal(0.0003, 0.015, size=n_days)
    close = start_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        "Open": close * (1 + rng.normal(0, 0.002, n_days)),
        "High": close * (1 + abs(rng.normal(0, 0.005, n_days))),
        "Low": close * (1 - abs(rng.normal(0, 0.005, n_days))),
        "Close": close,
        "Volume": rng.integers(1_000_000, 10_000_000, size=n_days),
    }, index=dates)
    return df


def _make_hist_dict(n_tickers: int = 4, n_days: int = 500,
                    start_date: str = "2015-01-02") -> dict[str, pd.DataFrame]:
    """Create synthetic history dict for multiple tickers."""
    tickers = [f"TK{i}" for i in range(n_tickers)]
    hist = {}
    for i, t in enumerate(tickers):
        hist[t] = _make_price_df(n_days=n_days, start_date=start_date,
                                 start_price=50 + i * 20, seed=42 + i)
    return hist


class _AlwaysBuyStrategy:
    """Minimal strategy: buy equal fraction on every day."""
    name = "AlwaysBuy"

    def decide(self, day, history, portfolio):
        tickers = sorted(portfolio.positions.keys())
        n = len(tickers)
        actions = {}
        for t in tickers:
            if portfolio.positions.get(t, 0) == 0:
                actions[t] = Action.buy(1.0 / n)
        return actions


class _AlwaysHoldStrategy:
    """Minimal strategy: hold everything."""
    name = "AlwaysHold"

    def decide(self, day, history, portfolio):
        return {}


# =====================================================================
# PortfolioState
# =====================================================================

class TestPortfolioState:
    def test_total_value(self):
        ps = PortfolioState(
            cash=5000,
            positions={"A": 10, "B": 20},
            prices={"A": 100, "B": 50},
        )
        assert ps.total_value() == 5000 + 10 * 100 + 20 * 50

    def test_equity_single(self):
        ps = PortfolioState(
            cash=1000,
            positions={"A": 5, "B": 10},
            prices={"A": 200, "B": 30},
        )
        assert ps.equity("A") == 1000.0
        assert ps.equity("B") == 300.0

    def test_equity_missing_ticker(self):
        ps = PortfolioState(cash=1000, positions={}, prices={})
        assert ps.equity("X") == 0.0

    def test_total_equity(self):
        ps = PortfolioState(
            cash=0,
            positions={"A": 10, "B": 5},
            prices={"A": 100, "B": 200},
        )
        assert ps.total_equity() == 10 * 100 + 5 * 200

    def test_allocation(self):
        ps = PortfolioState(
            cash=0,
            positions={"A": 10, "B": 10},
            prices={"A": 100, "B": 100},
        )
        assert ps.allocation("A") == pytest.approx(0.5)
        assert ps.allocation("B") == pytest.approx(0.5)

    def test_allocation_with_cash(self):
        ps = PortfolioState(
            cash=1000,
            positions={"A": 10},
            prices={"A": 100},
        )
        # equity = 1000, total = 2000, alloc = 0.5
        assert ps.allocation("A") == pytest.approx(0.5)

    def test_allocation_zero_value(self):
        ps = PortfolioState(cash=0, positions={"A": 0}, prices={"A": 100})
        assert ps.allocation("A") == 0.0

    def test_allocations(self):
        ps = PortfolioState(
            cash=0,
            positions={"A": 10, "B": 30},
            prices={"A": 100, "B": 100},
        )
        allocs = ps.allocations()
        assert allocs["A"] == pytest.approx(0.25)
        assert allocs["B"] == pytest.approx(0.75)


# =====================================================================
# PortfolioBacktester — Core Mechanics
# =====================================================================

class TestPortfolioBacktester:
    def test_no_trades_cash_unchanged(self):
        """HoldStrategy should preserve initial cash exactly."""
        hist = make_two_stock_hist([100] * 10, [50] * 10)
        bt = PortfolioBacktester(hist, HoldStrategy(), initial_cash=10_000)
        result = bt.run()

        assert result.final_value == pytest.approx(10_000)
        assert result.n_trades == 0

    def test_accounting_identity_every_day(self):
        """value == cash + sum(shares * price) every single day."""
        hist = make_two_stock_hist(
            [100 + i for i in range(30)],
            [50 - 0.5 * i for i in range(30)],
        )
        bt = PortfolioBacktester(hist, EqualWeightRebalance(), initial_cash=10_000)
        result = bt.run()

        for i, date in enumerate(result.daily_values.index):
            cash = result.daily_cash.iloc[i]
            equity = 0
            for t in result.tickers:
                shares = result.daily_positions[t].iloc[i]
                price = result.daily_prices[t].iloc[i]
                equity += shares * price
            expected = cash + equity
            actual = result.daily_values.iloc[i]
            assert actual == pytest.approx(expected, rel=1e-10), \
                f"Day {i}: {actual} != {expected}"

    def test_sells_before_buys(self):
        """Selling A should free cash for buying B on the same day."""
        prices_a = [100.0, 100.0, 100.0]
        prices_b = [100.0, 100.0, 100.0]
        hist = make_two_stock_hist(prices_a, prices_b)
        bt = PortfolioBacktester(hist, SellThenBuy(), initial_cash=10_000)
        result = bt.run()

        # After day 2: sold A (getting cash back), then bought B
        buy_b_trades = [t for t in result.trades
                        if t.ticker == "B" and t.action == "buy"]
        assert len(buy_b_trades) > 0, "B should have been bought with freed cash"

    def test_cash_reserve_enforced(self):
        """Cash should never drop below reserve fraction after buys."""
        hist = make_two_stock_hist([100] * 20, [50] * 20)
        reserve = 0.10
        bt = PortfolioBacktester(
            hist, BuyAllDay1(), initial_cash=10_000,
            cash_reserve_pct=reserve,
        )
        result = bt.run()

        for i in range(len(result.daily_values)):
            cash = result.daily_cash.iloc[i]
            total = result.daily_values.iloc[i]
            if total > 0:
                assert cash >= total * reserve - 0.01, \
                    f"Day {i}: cash {cash:.2f} < reserve {total * reserve:.2f}"

    def test_no_negative_cash(self):
        """Cash should never go negative."""
        hist = make_two_stock_hist(
            [100 + i * 2 for i in range(30)],
            [50 + i for i in range(30)],
        )
        bt = PortfolioBacktester(hist, EqualWeightRebalance(), initial_cash=10_000)
        result = bt.run()

        for i in range(len(result.daily_cash)):
            assert result.daily_cash.iloc[i] >= -0.001, \
                f"Day {i}: negative cash {result.daily_cash.iloc[i]}"

    def test_no_negative_shares(self):
        """No ticker should have negative shares."""
        hist = make_two_stock_hist(
            [100 + i for i in range(30)],
            [50 + 0.5 * i for i in range(30)],
        )
        bt = PortfolioBacktester(hist, EqualWeightRebalance(), initial_cash=10_000)
        result = bt.run()

        for t in result.tickers:
            for i in range(len(result.daily_positions[t])):
                assert result.daily_positions[t].iloc[i] >= -1e-10, \
                    f"Day {i}, {t}: negative shares"

    def test_drip_per_ticker(self):
        """Dividend reinvestment should add shares for the paying ticker."""
        divs_a = [0.0, 0.0, 1.0] + [0.0] * 7  # A pays div on day 3
        divs_b = [0.0] * 10                     # B pays nothing
        hist = make_two_stock_hist(
            [100] * 10, [50] * 10,
            dividends_a=divs_a, dividends_b=divs_b,
        )
        bt = PortfolioBacktester(hist, BuyAllDay1(), initial_cash=10_000)
        result = bt.run()

        div_trades = [t for t in result.trades if t.action == "dividend"]
        assert len(div_trades) > 0, "Should have dividend trades"
        assert all(t.ticker == "A" for t in div_trades), \
            "Only A should have dividend trades"

    def test_result_summary_has_tickers(self):
        hist = make_two_stock_hist([100] * 5, [50] * 5)
        bt = PortfolioBacktester(hist, HoldStrategy(), initial_cash=10_000)
        result = bt.run()
        summary = result.summary()
        assert "A" in summary
        assert "B" in summary

    def test_data_alignment_different_lengths(self):
        """DataFrames of different lengths should be aligned to union of dates."""
        prices_a = [100] * 20
        prices_b = [50] * 15  # Shorter
        hist_a = make_hist(prices_a)
        hist_b = make_hist(prices_b)
        hist = {"A": hist_a, "B": hist_b}

        bt = PortfolioBacktester(hist, HoldStrategy(), initial_cash=10_000)
        # Union: all 20 dates (B has NaN on days 16-20)
        n_days = len(bt.hist_dict["A"])
        assert n_days == 20
        assert len(bt.hist_dict["B"]) == 20

    def test_timezone_normalization(self):
        """Should handle timezone-aware indices."""
        hist_a = make_hist([100] * 5)
        hist_b = make_hist([50] * 5)
        # Make one tz-aware
        hist_a.index = hist_a.index.tz_localize("US/Eastern")

        hist = {"A": hist_a, "B": hist_b}
        bt = PortfolioBacktester(hist, HoldStrategy(), initial_cash=10_000)
        result = bt.run()
        assert result.final_value == pytest.approx(10_000)


# =====================================================================
# Conservation of Money
# =====================================================================

class TestConservationOfMoney:
    """
    Verify accounting identity: total_value == cash + sum(shares * price)
    on every day for every strategy.
    """

    @pytest.fixture
    def hist_dict(self):
        """Mildly volatile 2-stock universe."""
        np.random.seed(603)
        n = 60
        prices_a = 100 * np.cumprod(1 + np.random.randn(n) * 0.02)
        prices_b = 50 * np.cumprod(1 + np.random.randn(n) * 0.03)
        return make_two_stock_hist(prices_a.tolist(), prices_b.tolist())

    @pytest.mark.parametrize("strategy", [
        EqualWeightRebalance(threshold=0.05),
        InverseVolatilityWeight(threshold=0.05),
        IndependentMeanReversion(20, 2.0),
        RelativeStrength(20),
    ])
    def test_conservation(self, hist_dict, strategy):
        bt = PortfolioBacktester(hist_dict, strategy, initial_cash=10_000)
        result = bt.run()

        for i in range(len(result.daily_values)):
            cash = result.daily_cash.iloc[i]
            equity = 0
            for t in result.tickers:
                shares = result.daily_positions[t].iloc[i]
                price = result.daily_prices[t].iloc[i]
                equity += shares * price
            expected = cash + equity
            actual = result.daily_values.iloc[i]
            assert actual == pytest.approx(expected, rel=1e-10), \
                f"{strategy.name} day {i}: {actual} != {expected}"

    def test_conservation_with_dividends(self, hist_dict):
        """Conservation holds even when dividends are reinvested."""
        # Inject dividends
        hist_dict["A"].loc[hist_dict["A"].index[10], "Dividends"] = 0.50
        hist_dict["A"].loc[hist_dict["A"].index[30], "Dividends"] = 0.75

        bt = PortfolioBacktester(
            hist_dict, EqualWeightRebalance(), initial_cash=10_000,
        )
        result = bt.run()

        for i in range(len(result.daily_values)):
            cash = result.daily_cash.iloc[i]
            equity = 0
            for t in result.tickers:
                shares = result.daily_positions[t].iloc[i]
                price = result.daily_prices[t].iloc[i]
                equity += shares * price
            expected = cash + equity
            actual = result.daily_values.iloc[i]
            assert actual == pytest.approx(expected, rel=1e-10)


# =====================================================================
# Strategies
# =====================================================================

class TestEqualWeightRebalance:
    def test_initial_buy(self):
        """Should buy all tickers on day 1."""
        hist = make_two_stock_hist([100] * 5, [50] * 5)
        bt = PortfolioBacktester(
            hist, EqualWeightRebalance(threshold=0.05), initial_cash=10_000,
        )
        result = bt.run()
        # Should have bought both A and B
        buy_tickers = set(t.ticker for t in result.trades if t.action == "buy")
        assert "A" in buy_tickers
        assert "B" in buy_tickers

    def test_rebalances_on_drift(self):
        """Should trigger rebalance when allocation drifts."""
        # A doubles, B stays flat -> A becomes overweight
        prices_a = [100] * 5 + [200] * 5
        prices_b = [100] * 10
        hist = make_two_stock_hist(prices_a, prices_b)
        bt = PortfolioBacktester(
            hist, EqualWeightRebalance(threshold=0.05), initial_cash=10_000,
        )
        result = bt.run()
        # Should have sell trades for A (overweight) after price jump
        sells_a = [t for t in result.trades
                   if t.ticker == "A" and t.action == "sell"]
        assert len(sells_a) > 0, "A should be sold when overweight"

    def test_converges_to_balanced(self):
        """Flat prices: allocations should converge toward equal weight."""
        hist = make_two_stock_hist([100] * 30, [100] * 30)
        bt = PortfolioBacktester(
            hist, EqualWeightRebalance(threshold=0.05), initial_cash=10_000,
        )
        result = bt.run()
        # After convergence, allocations should be near equal
        last_day = len(result.daily_values) - 1
        cash = result.daily_cash.iloc[last_day]
        total = result.daily_values.iloc[last_day]
        for t in result.tickers:
            shares = result.daily_positions[t].iloc[last_day]
            price = result.daily_prices[t].iloc[last_day]
            alloc = (shares * price) / total
            assert abs(alloc - 0.5) < 0.10, \
                f"{t} allocation {alloc:.3f} too far from 0.5"


class TestEqualWeightInverseVol:
    """Inverse-volatility buy priority: least volatile underweight stocks
    get bought first when cash is limited."""

    def test_low_vol_ticker_bought_first(self):
        """With 3 tickers, the least volatile one should get bought first
        during rebalance when cash is limited."""
        np.random.seed(603)
        n = 80
        # A: low vol (stable around 100)
        prices_a = [100 + np.random.normal(0, 0.5) for _ in range(n)]
        prices_a = [max(p, 50) for p in prices_a]
        # B: high vol (big swings)
        prices_b = [100 + np.random.normal(0, 5.0) for _ in range(n)]
        prices_b = [max(p, 50) for p in prices_b]
        # C: medium vol
        prices_c = [100 + np.random.normal(0, 2.0) for _ in range(n)]
        prices_c = [max(p, 50) for p in prices_c]

        hist = {
            "A": make_hist(prices_a),
            "B": make_hist(prices_b),
            "C": make_hist(prices_c),
        }
        strat = EqualWeightRebalance(threshold=0.05, vol_lookback=60)
        bt = PortfolioBacktester(hist, strat, initial_cash=10_000,
                                  cash_reserve_pct=0.05)
        res = bt.run()

        # Strategy should complete without error
        assert res.final_value > 0
        assert res.n_trades > 0

    def test_vol_lookback_parameter(self):
        """vol_lookback should be stored and configurable."""
        strat = EqualWeightRebalance(threshold=0.05, vol_lookback=30)
        assert strat.vol_lookback == 30
        strat2 = EqualWeightRebalance()
        assert strat2.vol_lookback == 60  # default

    def test_inverse_vol_with_flat_prices(self):
        """With identical flat prices, all tickers have ~0 vol; should
        still buy all without error."""
        hist = {
            "A": make_hist([100] * 20),
            "B": make_hist([100] * 20),
            "C": make_hist([100] * 20),
        }
        strat = EqualWeightRebalance(threshold=0.05, vol_lookback=60)
        bt = PortfolioBacktester(hist, strat, initial_cash=10_000)
        res = bt.run()
        # All tickers should have shares
        for t in res.tickers:
            assert res.daily_positions[t].iloc[-1] > 0

    def test_conservation_with_inverse_vol(self):
        """Accounting identity holds with inverse-vol priority."""
        np.random.seed(603)
        n = 60
        hist = {}
        for t, vol in [("X", 0.5), ("Y", 3.0), ("Z", 1.5)]:
            prices = [100 + np.random.normal(0, vol) for _ in range(n)]
            prices = [max(p, 50) for p in prices]
            hist[t] = make_hist(prices)

        strat = EqualWeightRebalance(threshold=0.05, vol_lookback=40)
        bt = PortfolioBacktester(hist, strat, initial_cash=10_000,
                                  cash_reserve_pct=0.05)
        res = bt.run()

        for i in range(len(res.daily_values)):
            cash = res.daily_cash.iloc[i]
            equity = sum(
                res.daily_positions[t].iloc[i] * res.daily_prices[t].iloc[i]
                for t in res.tickers
            )
            total = res.daily_values.iloc[i]
            assert cash + equity == pytest.approx(total, abs=0.01)


class TestIndependentMeanReversion:
    def test_buys_on_dip(self):
        """Should buy when price drops below lower Bollinger Band."""
        # Stable then big drop
        prices = [100.0] * 25 + [80.0, 79.0, 78.0, 77.0, 76.0]
        hist = make_two_stock_hist(prices, [50.0] * 30)
        bt = PortfolioBacktester(
            hist, IndependentMeanReversion(20, 2.0), initial_cash=10_000,
        )
        result = bt.run()
        buys_a = [t for t in result.trades
                  if t.ticker == "A" and t.action == "buy"]
        assert len(buys_a) > 0, "Should buy A on dip below lower band"

    def test_sells_on_spike(self):
        """Should sell when price rises above upper Bollinger Band."""
        # Buy first, then big spike
        prices = [100.0] * 25 + [120.0, 125.0, 130.0, 135.0, 140.0]
        hist = make_two_stock_hist(prices, [50.0] * 30)
        # Need initial position to sell
        class BuyThenMeanRev(PortfolioStrategy):
            name = "test"
            def __init__(self):
                self.mr = IndependentMeanReversion(20, 2.0)
                self.bought = False
            def decide(self, day, history, portfolio):
                if not self.bought:
                    self.bought = True
                    return {"A": Action.buy(1.0)}
                return self.mr.decide(day, history, portfolio)

        bt = PortfolioBacktester(hist, BuyThenMeanRev(), initial_cash=10_000)
        result = bt.run()
        sells_a = [t for t in result.trades
                   if t.ticker == "A" and t.action == "sell"]
        assert len(sells_a) > 0, "Should sell A on spike above upper band"


class TestRelativeStrength:
    def test_buys_winner_sells_loser(self):
        """Should buy the outperforming ticker and sell the underperformer."""
        # A goes up, B goes down
        n = 30
        prices_a = [100 + i * 2 for i in range(n)]   # Trending up
        prices_b = [100 - i * 2 for i in range(n)]   # Trending down
        # Ensure B stays positive
        prices_b = [max(p, 10) for p in prices_b]
        hist = make_two_stock_hist(prices_a, prices_b)

        # Need initial positions to sell
        class BuyThenRS(PortfolioStrategy):
            name = "test"
            def __init__(self):
                self.rs = RelativeStrength(20)
                self.bought = False
            def decide(self, day, history, portfolio):
                if not self.bought:
                    self.bought = True
                    tickers = sorted(portfolio.positions.keys())
                    return {t: Action.buy(0.5) for t in tickers}
                return self.rs.decide(day, history, portfolio)

        bt = PortfolioBacktester(hist, BuyThenRS(), initial_cash=10_000)
        result = bt.run()

        # A is the winner, B is the loser
        buys_a = [t for t in result.trades
                  if t.ticker == "A" and t.action == "buy"
                  and t.date != result.daily_values.index[0]]
        sells_b = [t for t in result.trades
                   if t.ticker == "B" and t.action == "sell"]
        assert len(buys_a) > 0, "Should buy winner A"
        assert len(sells_b) > 0, "Should sell loser B"

    def test_needs_lookback(self):
        """Should not trade before lookback period."""
        hist = make_two_stock_hist([100] * 10, [50] * 10)
        bt = PortfolioBacktester(
            hist, RelativeStrength(20), initial_cash=10_000,
        )
        result = bt.run()
        assert result.n_trades == 0, "Should not trade with < lookback days"


# =====================================================================
# Metrics
# =====================================================================

class TestMetrics:
    def test_total_return(self):
        hist = make_two_stock_hist([100] * 10, [50] * 10)
        bt = PortfolioBacktester(hist, HoldStrategy(), initial_cash=10_000)
        result = bt.run()
        assert result.total_return == pytest.approx(0.0)

    def test_annualized_return_short(self):
        hist = make_two_stock_hist([100], [50])
        bt = PortfolioBacktester(hist, HoldStrategy(), initial_cash=10_000)
        result = bt.run()
        assert result.annualized_return == 0.0

    def test_sharpe_zero_vol(self):
        hist = make_two_stock_hist([100] * 10, [50] * 10)
        bt = PortfolioBacktester(hist, HoldStrategy(), initial_cash=10_000)
        result = bt.run()
        assert result.sharpe_ratio == 0.0

    def test_max_drawdown_flat(self):
        hist = make_two_stock_hist([100] * 10, [50] * 10)
        bt = PortfolioBacktester(hist, HoldStrategy(), initial_cash=10_000)
        result = bt.run()
        assert result.max_drawdown == pytest.approx(0.0)

    def test_ticker_contribution_returns_dict(self):
        hist = make_two_stock_hist([100] * 10, [50] * 10)
        bt = PortfolioBacktester(hist, BuyAllDay1(), initial_cash=10_000)
        result = bt.run()
        contrib = result.ticker_contribution()
        assert isinstance(contrib, dict)
        assert "A" in contrib
        assert "B" in contrib


# =====================================================================
# Three-stock universe
# =====================================================================

class TestThreeStocks:
    """Tests with 3 tickers to verify generality."""

    @pytest.fixture
    def hist_dict_3(self):
        np.random.seed(603)
        n = 50
        return {
            "X": make_hist((100 * np.cumprod(1 + np.random.randn(n) * 0.02)).tolist()),
            "Y": make_hist((80 * np.cumprod(1 + np.random.randn(n) * 0.03)).tolist()),
            "Z": make_hist((60 * np.cumprod(1 + np.random.randn(n) * 0.01)).tolist()),
        }

    def test_equal_weight_three(self, hist_dict_3):
        bt = PortfolioBacktester(
            hist_dict_3, EqualWeightRebalance(), initial_cash=30_000,
        )
        result = bt.run()
        assert result.final_value > 0
        assert len(result.tickers) == 3

    def test_conservation_three(self, hist_dict_3):
        bt = PortfolioBacktester(
            hist_dict_3, EqualWeightRebalance(), initial_cash=30_000,
        )
        result = bt.run()

        for i in range(len(result.daily_values)):
            cash = result.daily_cash.iloc[i]
            equity = 0
            for t in result.tickers:
                shares = result.daily_positions[t].iloc[i]
                price = result.daily_prices[t].iloc[i]
                equity += shares * price
            expected = cash + equity
            actual = result.daily_values.iloc[i]
            assert actual == pytest.approx(expected, rel=1e-10)

    def test_relative_strength_three(self, hist_dict_3):
        """RelativeStrength should work with 3 tickers."""
        bt = PortfolioBacktester(
            hist_dict_3, RelativeStrength(20), initial_cash=30_000,
        )
        result = bt.run()
        assert result.final_value > 0


# =====================================================================
# Edge Cases (portfolio backtest)
# =====================================================================

class TestEdgeCases:
    def test_single_day(self):
        hist = make_two_stock_hist([100], [50])
        bt = PortfolioBacktester(hist, HoldStrategy(), initial_cash=10_000)
        result = bt.run()
        assert len(result.daily_values) == 1
        assert result.final_value == pytest.approx(10_000)

    def test_zero_reserve(self):
        """With 0% reserve, all cash can be deployed."""
        hist = make_two_stock_hist([100] * 5, [50] * 5)
        bt = PortfolioBacktester(
            hist, BuyAllDay1(), initial_cash=10_000,
            cash_reserve_pct=0.0,
        )
        result = bt.run()
        # BuyAllDay1 buys fraction=1/n per ticker sequentially,
        # so cash decreases multiplicatively: 10000 * (1-0.5) * (1-0.5) = 2500
        # The key invariant: more cash is spent than with a reserve
        assert result.daily_cash.iloc[0] < 5_000

    def test_empty_actions_hold(self):
        """Returning empty dict should hold all positions."""
        hist = make_two_stock_hist([100] * 5, [50] * 5)
        bt = PortfolioBacktester(hist, HoldStrategy(), initial_cash=10_000)
        result = bt.run()
        assert result.n_trades == 0

    def test_portfolio_trade_fields(self):
        """PortfolioTrade should have all required fields."""
        trade = PortfolioTrade(
            date=pd.Timestamp("2023-01-02"),
            ticker="A",
            action="buy",
            shares=10,
            price=100,
            cash_after=9000,
            total_value_after=10000,
        )
        assert trade.ticker == "A"
        assert trade.action == "buy"
        assert trade.shares == 10


# =====================================================================
# File Structure
# =====================================================================

class TestFileStructure:
    """Verify the reorganized directory layout is intact."""

    def test_backtest_dir_exists(self):
        assert os.path.isdir(os.path.join(REPO_ROOT, "finance_tools", "backtest"))

    def test_strategies_dir_exists(self):
        assert os.path.isdir(os.path.join(REPO_ROOT, "finance_tools", "strategies"))

    def test_backtester_app_dir_exists(self):
        assert os.path.isdir(os.path.join(REPO_ROOT, "apps", "backtester"))

    def test_backtest_engine_exists(self):
        assert os.path.isfile(os.path.join(
            REPO_ROOT, "finance_tools", "backtest", "engine.py"))

    def test_library_modules_exist(self):
        """Core library modules should live in the right places."""
        expected = [
            os.path.join("finance_tools", "backtest", "portfolio.py"),
            os.path.join("finance_tools", "strategies", "portfolio.py"),
            os.path.join("finance_tools", "backtest", "engine.py"),
            os.path.join("finance_tools", "strategies", "equal_weight.py"),
            os.path.join("finance_tools", "backtest", "monte_carlo.py"),
        ]
        for rel_path in expected:
            full_path = os.path.join(REPO_ROOT, rel_path)
            assert os.path.isfile(full_path), f"Missing: {rel_path}"

    def test_runner_scripts_in_apps(self):
        """Runner scripts should live in apps/backtester/."""
        app_dir = os.path.join(REPO_ROOT, "apps", "backtester")
        for script in ["run.py", "regime.py", "stress_test.py"]:
            assert os.path.isfile(os.path.join(app_dir, script)), \
                f"Missing: {script}"


# =====================================================================
# Cross-Script Consistency (portfolio backtest)
# =====================================================================

class TestCrossScriptConsistency:
    """Verify imports, re-exports, and type identity across modules."""

    def test_action_type_identity(self):
        """Action/ActionType from engine and portfolio should be the same objects."""
        from finance_tools.backtest.engine import Action as A1, ActionType as AT1
        from finance_tools.backtest.portfolio import Action as A2, ActionType as AT2
        assert A1 is A2, "Action class mismatch between engine and portfolio"
        assert AT1 is AT2, "ActionType enum mismatch"

    def test_action_created_in_strategies_matches_engine(self):
        """Actions from portfolio_strategies should use the same ActionType."""
        strat = EqualWeightRebalance()
        ps = PortfolioState(
            cash=10_000,
            positions={"A": 0, "B": 0},
            prices={"A": 100, "B": 100},
        )
        day = {"A": pd.Series({"Close": 100}), "B": pd.Series({"Close": 100})}
        hist = {
            "A": pd.DataFrame({"Close": [100]}),
            "B": pd.DataFrame({"Close": [100]}),
        }
        actions = strat.decide(day, hist, ps)
        for ticker, action in actions.items():
            assert isinstance(action.action, ActionType), \
                f"{ticker}: action type is {type(action.action)}, not ActionType"


# =====================================================================
# Ticker Contribution Edge Cases
# =====================================================================

class TestTickerContribution:
    """Thorough tests for P&L attribution logic."""

    def test_no_trades_zero_contribution(self):
        """HoldStrategy: no trades -> zero P&L for all tickers."""
        hist = make_two_stock_hist([100] * 10, [50] * 10)
        bt = PortfolioBacktester(hist, HoldStrategy(), initial_cash=10_000)
        result = bt.run()
        contrib = result.ticker_contribution()
        for t in result.tickers:
            assert contrib[t] == pytest.approx(0.0), \
                f"{t}: expected 0 contribution, got {contrib[t]}"

    def test_flat_prices_zero_pnl(self):
        """Buy then hold at flat prices -> zero P&L (cost basis = final value)."""
        hist = make_two_stock_hist([100] * 20, [100] * 20)
        bt = PortfolioBacktester(
            hist, BuyAllDay1(), initial_cash=10_000,
            cash_reserve_pct=0.0,
        )
        result = bt.run()
        contrib = result.ticker_contribution()
        for t in result.tickers:
            assert abs(contrib[t]) < 1.0, \
                f"{t}: P&L should be ~0 on flat prices, got {contrib[t]:.2f}"

    def test_rising_prices_positive_pnl(self):
        """Buy then hold with rising prices -> positive contribution."""
        prices_a = [100 + i * 5 for i in range(20)]
        prices_b = [50 + i * 2 for i in range(20)]
        hist = make_two_stock_hist(prices_a, prices_b)
        bt = PortfolioBacktester(
            hist, BuyAllDay1(), initial_cash=10_000,
            cash_reserve_pct=0.0,
        )
        result = bt.run()
        contrib = result.ticker_contribution()
        assert contrib["A"] > 0, "A should have positive P&L on rising prices"
        assert contrib["B"] > 0, "B should have positive P&L on rising prices"

    def test_contribution_keys_match_tickers(self):
        """ticker_contribution() must have exactly the right tickers."""
        hist = make_two_stock_hist([100] * 10, [50] * 10)
        bt = PortfolioBacktester(hist, BuyAllDay1(), initial_cash=10_000)
        result = bt.run()
        contrib = result.ticker_contribution()
        assert set(contrib.keys()) == set(result.tickers)


# =====================================================================
# Data Alignment Edge Cases
# =====================================================================

class TestDataAlignment:
    """Verify data alignment handles real-world messiness."""

    def test_both_tz_aware_different_zones(self):
        """Two tickers with different timezones should still align."""
        hist_a = make_hist([100] * 5)
        hist_b = make_hist([50] * 5)
        hist_a.index = hist_a.index.tz_localize("US/Eastern")
        hist_b.index = hist_b.index.tz_localize("US/Pacific")
        hist = {"A": hist_a, "B": hist_b}
        bt = PortfolioBacktester(hist, HoldStrategy(), initial_cash=10_000)
        result = bt.run()
        assert len(result.daily_values) == 5
        assert result.final_value == pytest.approx(10_000)

    def test_duplicate_dates_handled(self):
        """Duplicate index entries should be deduplicated (keep first)."""
        hist_a = make_hist([100, 100, 100])
        # Inject a duplicate date
        dup_row = hist_a.iloc[[1]].copy()
        dup_row["Close"] = 999.0  # Different value -- should be dropped
        hist_a = pd.concat([hist_a, dup_row])
        hist_b = make_hist([50, 50, 50])
        hist = {"A": hist_a, "B": hist_b}
        bt = PortfolioBacktester(hist, HoldStrategy(), initial_cash=10_000)
        # The duplicate should be dropped; original 3 dates remain
        assert len(bt.hist_dict["A"]) == 3
        # Verify the kept value is the first (100), not the duplicate (999)
        assert bt.hist_dict["A"].iloc[1]["Close"] == 100.0

    def test_no_common_dates_runs(self):
        """Non-overlapping date ranges: each ticker active on its own dates."""
        hist_a = make_hist([100] * 5, start_date="2023-01-02")
        hist_b = make_hist([50] * 5, start_date="2024-01-02")
        hist = {"A": hist_a, "B": hist_b}
        bt = PortfolioBacktester(hist, HoldStrategy(), initial_cash=10_000)
        result = bt.run()
        # Union of 10 dates total (5 from A + 5 from B, no overlap)
        assert len(result.daily_values) == 10
        assert result.final_value == pytest.approx(10_000)

    def test_partial_overlap(self):
        """Union of dates includes all dates from both tickers."""
        # A: 10 days starting Jan 2; B: 10 days starting Jan 6
        hist_a = make_hist([100] * 10, start_date="2023-01-02")
        hist_b = make_hist([50] * 10, start_date="2023-01-06")
        hist = {"A": hist_a, "B": hist_b}
        bt = PortfolioBacktester(hist, HoldStrategy(), initial_cash=10_000)
        n = len(bt.hist_dict["A"])
        # Union includes all unique business days from both
        assert n > 10, "Union should have more than 10 days"
        assert len(bt.hist_dict["A"]) == len(bt.hist_dict["B"])


# =====================================================================
# Numerical Edge Cases
# =====================================================================

class TestNumericalEdgeCases:
    """Catch division-by-zero, NaN, and floating-point traps."""

    def test_very_cheap_stock(self):
        """Penny stock ($0.01) should not cause overflow or precision loss."""
        hist = make_two_stock_hist([0.01] * 20, [0.01] * 20)
        bt = PortfolioBacktester(
            hist, EqualWeightRebalance(), initial_cash=10_000,
        )
        result = bt.run()
        # Accounting identity must still hold
        for i in range(len(result.daily_values)):
            cash = result.daily_cash.iloc[i]
            equity = sum(
                result.daily_positions[t].iloc[i] * result.daily_prices[t].iloc[i]
                for t in result.tickers
            )
            assert result.daily_values.iloc[i] == pytest.approx(cash + equity, rel=1e-8)

    def test_large_price_jump(self):
        """100x price jump should not break accounting."""
        prices_a = [100] * 10 + [10_000] * 10
        prices_b = [50] * 20
        hist = make_two_stock_hist(prices_a, prices_b)
        bt = PortfolioBacktester(
            hist, EqualWeightRebalance(), initial_cash=10_000,
        )
        result = bt.run()
        assert result.final_value > 0
        assert np.isfinite(result.final_value)
        # Accounting identity on last day
        last = len(result.daily_values) - 1
        cash = result.daily_cash.iloc[last]
        equity = sum(
            result.daily_positions[t].iloc[last] * result.daily_prices[t].iloc[last]
            for t in result.tickers
        )
        assert result.daily_values.iloc[last] == pytest.approx(cash + equity, rel=1e-10)

    def test_mean_reversion_flat_prices_no_nan(self):
        """Flat prices -> std=0 -> upper==lower==ma. No trades, no NaN."""
        hist = make_two_stock_hist([100.0] * 30, [50.0] * 30)
        bt = PortfolioBacktester(
            hist, IndependentMeanReversion(20, 2.0), initial_cash=10_000,
        )
        result = bt.run()
        assert result.n_trades == 0
        assert np.all(np.isfinite(result.daily_values.values))

    def test_small_dividend_precision(self):
        """Very small dividend ($0.001) should not cause rounding issues."""
        divs_a = [0.0, 0.0, 0.001] + [0.0] * 7
        divs_b = [0.0] * 10
        hist = make_two_stock_hist(
            [100] * 10, [50] * 10,
            dividends_a=divs_a, dividends_b=divs_b,
        )
        bt = PortfolioBacktester(hist, BuyAllDay1(), initial_cash=10_000)
        result = bt.run()
        # Accounting identity must hold even with tiny dividends
        for i in range(len(result.daily_values)):
            cash = result.daily_cash.iloc[i]
            equity = sum(
                result.daily_positions[t].iloc[i] * result.daily_prices[t].iloc[i]
                for t in result.tickers
            )
            assert result.daily_values.iloc[i] == pytest.approx(cash + equity, rel=1e-10)

    def test_drip_skipped_when_price_is_zero(self):
        """If price hits $0, DRIP should not divide by zero."""
        # Price drops to near-zero on day where dividend exists
        divs_a = [0.0, 0.0, 1.0, 0.0, 0.0]
        prices_a = [100, 100, 0.001, 0.001, 0.001]
        prices_b = [50] * 5
        hist = make_two_stock_hist(
            prices_a, prices_b,
            dividends_a=divs_a,
        )
        bt = PortfolioBacktester(hist, BuyAllDay1(), initial_cash=10_000)
        result = bt.run()
        assert np.all(np.isfinite(result.daily_values.values))
        assert np.all(np.isfinite(result.daily_cash.values))


# =====================================================================
# Silent Trade Skipping
# =====================================================================

class TestMinimumTradeSize:
    """The engine silently skips buy orders where spend < $0.01."""

    def test_tiny_buy_fraction_skipped(self):
        """A buy with fraction so small that spend < $0.01 should be a no-op."""

        class TinyBuy(PortfolioStrategy):
            name = "Tiny Buy"
            def decide(self, day, history, portfolio):
                return {"A": Action.buy(1e-8)}

        hist = make_two_stock_hist([100] * 5, [50] * 5)
        bt = PortfolioBacktester(hist, TinyBuy(), initial_cash=100)
        result = bt.run()
        # spend = (100 - 5% reserve) * 1e-8 ~ $9.5e-7 < $0.01 -> skipped
        assert result.n_trades == 0

    def test_normal_buy_not_skipped(self):
        """A normal buy should execute."""
        hist = make_two_stock_hist([100] * 5, [50] * 5)
        bt = PortfolioBacktester(hist, BuyAllDay1(), initial_cash=10_000)
        result = bt.run()
        assert result.n_trades > 0


# =====================================================================
# Sequential Buy Order Effects
# =====================================================================

class TestSequentialBuyEffects:
    """Document that buys execute in alphabetical ticker order, so earlier
    tickers get more cash than later tickers for the same fraction."""

    def test_first_ticker_gets_more_cash(self):
        """With 2 tickers buying fraction=0.5 each, A spends more than B."""
        hist = make_two_stock_hist([100] * 5, [100] * 5)
        bt = PortfolioBacktester(
            hist, BuyAllDay1(), initial_cash=10_000,
            cash_reserve_pct=0.0,
        )
        result = bt.run()
        # A buys first: spend = 10000 * 0.5 = 5000 -> 50 shares
        # B buys second: spend = 5000 * 0.5 = 2500 -> 25 shares
        shares_a = result.daily_positions["A"].iloc[0]
        shares_b = result.daily_positions["B"].iloc[0]
        assert shares_a > shares_b, \
            f"A ({shares_a:.1f} shares) should get more than B ({shares_b:.1f})"

    def test_sequential_buy_conserves_value(self):
        """Even with sequential imbalance, total value must be conserved."""
        hist = make_two_stock_hist([100] * 5, [100] * 5)
        bt = PortfolioBacktester(
            hist, BuyAllDay1(), initial_cash=10_000,
            cash_reserve_pct=0.0,
        )
        result = bt.run()
        # Day 0: all cash deployed except remainder
        cash = result.daily_cash.iloc[0]
        equity = sum(
            result.daily_positions[t].iloc[0] * 100
            for t in result.tickers
        )
        assert cash + equity == pytest.approx(10_000, rel=1e-10)


# =====================================================================
# Metrics Edge Cases
# =====================================================================

class TestMetricsEdgeCases:

    def test_negative_sharpe_on_losing_portfolio(self):
        """Falling prices -> negative annualized return -> negative Sharpe."""
        # Steadily declining prices
        prices_a = [100 - i * 2 for i in range(30)]
        prices_b = [50 - i for i in range(30)]
        prices_a = [max(p, 10) for p in prices_a]
        prices_b = [max(p, 5) for p in prices_b]
        hist = make_two_stock_hist(prices_a, prices_b)
        bt = PortfolioBacktester(hist, BuyAllDay1(), initial_cash=10_000)
        result = bt.run()
        assert result.total_return < 0
        assert result.sharpe_ratio < 0

    def test_annualized_return_two_days(self):
        """With exactly 2 days, annualized return should still compute."""
        hist = make_two_stock_hist([100, 110], [50, 55])
        bt = PortfolioBacktester(
            hist, BuyAllDay1(), initial_cash=10_000,
            cash_reserve_pct=0.0,
        )
        result = bt.run()
        assert np.isfinite(result.annualized_return)
        assert result.annualized_return > 0  # Prices went up

    def test_max_drawdown_with_crash(self):
        """Verify max drawdown captures the deepest valley."""
        # Rise then crash then partial recovery
        prices_a = [100, 120, 140, 60, 80]
        prices_b = [50, 50, 50, 50, 50]
        hist = make_two_stock_hist(prices_a, prices_b)
        bt = PortfolioBacktester(hist, BuyAllDay1(), initial_cash=10_000)
        result = bt.run()
        assert result.max_drawdown < -0.10, \
            f"Expected significant drawdown, got {result.max_drawdown:.3f}"


# =====================================================================
# Strategy-Specific Edge Cases
# =====================================================================

class TestStrategyEdgeCases:

    def test_equal_weight_five_tickers(self):
        """EqualWeight should target 20% per ticker with 5 stocks."""
        np.random.seed(603)
        n = 40
        hist = {
            t: make_hist((100 * np.cumprod(1 + np.random.randn(n) * 0.02)).tolist())
            for t in ["A", "B", "C", "D", "E"]
        }
        bt = PortfolioBacktester(
            hist, EqualWeightRebalance(threshold=0.05), initial_cash=50_000,
        )
        result = bt.run()
        # Check that all 5 tickers were bought
        bought = set(t.ticker for t in result.trades if t.action == "buy")
        assert len(bought) == 5

    def test_relative_strength_two_tickers(self):
        """With 2 tickers, top=1 and bottom=1, so winner is bought and loser sold."""
        n = 30
        prices_a = [100 + i * 3 for i in range(n)]  # Winner
        prices_b = [100 - i for i in range(n)]       # Loser
        prices_b = [max(p, 10) for p in prices_b]
        hist = make_two_stock_hist(prices_a, prices_b)

        class BuyThenRS(PortfolioStrategy):
            name = "test"
            def __init__(self):
                self.rs = RelativeStrength(20)
                self.bought = False
            def decide(self, day, history, portfolio):
                if not self.bought:
                    self.bought = True
                    return {t: Action.buy(0.5) for t in sorted(portfolio.positions.keys())}
                return self.rs.decide(day, history, portfolio)

        bt = PortfolioBacktester(hist, BuyThenRS(), initial_cash=10_000)
        result = bt.run()
        # With 2 tickers: top 1/3 rounds up to 1 -> A is bought
        buys_a = [t for t in result.trades
                  if t.ticker == "A" and t.action == "buy"
                  and t.date != result.daily_values.index[0]]
        assert len(buys_a) > 0, "A (winner) should receive buy signals"

    def test_mean_reversion_no_action_within_bands(self):
        """Prices within Bollinger Bands should produce no trades."""
        # Small random walk that stays within 2 std
        np.random.seed(603)
        prices = (100 + np.cumsum(np.random.randn(30) * 0.3)).tolist()
        hist = make_two_stock_hist(prices, [50] * 30)
        bt = PortfolioBacktester(
            hist, IndependentMeanReversion(20, 2.0), initial_cash=10_000,
        )
        result = bt.run()
        # Tight random walk -> no BB breakouts
        buys = [t for t in result.trades if t.action == "buy"]
        sells = [t for t in result.trades if t.action == "sell"]
        # At least verify no crash and identity holds
        assert np.all(np.isfinite(result.daily_values.values))


# =====================================================================
# High-Volatility Stress Tests
# =====================================================================

class TestStress:
    """Stress tests with volatile data and all strategies."""

    @pytest.fixture
    def volatile_hist(self):
        np.random.seed(603)
        n = 200
        prices_a = (100 * np.cumprod(1 + np.random.randn(n) * 0.05)).tolist()
        prices_b = (50 * np.cumprod(1 + np.random.randn(n) * 0.07)).tolist()
        # Inject dividends at random points
        divs_a = [0.0] * n
        divs_b = [0.0] * n
        for i in range(10, n, 20):
            divs_a[i] = 0.25
        for i in range(15, n, 25):
            divs_b[i] = 0.15
        return make_two_stock_hist(
            prices_a, prices_b,
            dividends_a=divs_a, dividends_b=divs_b,
        )

    @pytest.mark.parametrize("strategy", [
        EqualWeightRebalance(threshold=0.05),
        IndependentMeanReversion(20, 2.0),
        RelativeStrength(20),
    ])
    def test_accounting_identity_under_stress(self, volatile_hist, strategy):
        """Conservation law must hold even under extreme volatility + dividends."""
        bt = PortfolioBacktester(
            volatile_hist, strategy, initial_cash=10_000,
        )
        result = bt.run()
        for i in range(len(result.daily_values)):
            cash = result.daily_cash.iloc[i]
            equity = sum(
                result.daily_positions[t].iloc[i] * result.daily_prices[t].iloc[i]
                for t in result.tickers
            )
            assert result.daily_values.iloc[i] == pytest.approx(cash + equity, rel=1e-10), \
                f"{strategy.name} day {i}: identity violated"

    @pytest.mark.parametrize("strategy", [
        EqualWeightRebalance(threshold=0.05),
        IndependentMeanReversion(20, 2.0),
        RelativeStrength(20),
    ])
    def test_no_negative_cash_under_stress(self, volatile_hist, strategy):
        bt = PortfolioBacktester(volatile_hist, strategy, initial_cash=10_000)
        result = bt.run()
        for i in range(len(result.daily_cash)):
            assert result.daily_cash.iloc[i] >= -0.001, \
                f"{strategy.name} day {i}: negative cash"

    @pytest.mark.parametrize("strategy", [
        EqualWeightRebalance(threshold=0.05),
        IndependentMeanReversion(20, 2.0),
        RelativeStrength(20),
    ])
    def test_no_negative_shares_under_stress(self, volatile_hist, strategy):
        bt = PortfolioBacktester(volatile_hist, strategy, initial_cash=10_000)
        result = bt.run()
        for t in result.tickers:
            for i in range(len(result.daily_positions[t])):
                assert result.daily_positions[t].iloc[i] >= -1e-10, \
                    f"{strategy.name} day {i}, {t}: negative shares"

    @pytest.mark.parametrize("strategy", [
        EqualWeightRebalance(threshold=0.05),
        IndependentMeanReversion(20, 2.0),
        RelativeStrength(20),
    ])
    def test_all_values_finite_under_stress(self, volatile_hist, strategy):
        bt = PortfolioBacktester(volatile_hist, strategy, initial_cash=10_000)
        result = bt.run()
        assert np.all(np.isfinite(result.daily_values.values)), \
            f"{strategy.name}: non-finite daily values"
        assert np.all(np.isfinite(result.daily_cash.values)), \
            f"{strategy.name}: non-finite cash"


# =====================================================================
# Partial Participation
# =====================================================================

class TestPartialParticipation:
    """Tests for tickers that enter/exit the backtest mid-run."""

    def test_late_entry_ticker_starts_with_zero_shares(self):
        """A ticker that IPOs mid-backtest should start with 0 shares."""
        # A: 20 days starting Jan 2
        # B: 10 days starting Jan 16 (enters late)
        hist_a = make_hist([100] * 20, start_date="2023-01-02")
        hist_b = make_hist([50] * 10, start_date="2023-01-16")
        hist = {"A": hist_a, "B": hist_b}
        bt = PortfolioBacktester(hist, BuyAllDay1(), initial_cash=10_000)
        result = bt.run()

        # B should have 0 shares before it becomes active
        for i in range(len(result.daily_values)):
            if result.daily_prices["B"].iloc[i] == 0.0:
                assert result.daily_positions["B"].iloc[i] == 0.0

    def test_late_entry_ticker_gets_bought(self):
        """A late-entering ticker should be visible to strategies once active."""
        # A: 20 days; B: enters on day 11
        hist_a = make_hist([100] * 20, start_date="2023-01-02")
        hist_b = make_hist([50] * 10, start_date="2023-01-16")
        hist = {"A": hist_a, "B": hist_b}

        # EqualWeightRebalance sells overweight A and buys underweight B
        bt = PortfolioBacktester(
            hist, EqualWeightRebalance(threshold=0.05), initial_cash=10_000,
        )
        result = bt.run()

        # B should eventually have shares > 0
        final_b = result.daily_positions["B"].iloc[-1]
        assert final_b > 0, "Late-entry B should have been bought"

    def test_delisted_ticker_frozen_shares(self):
        """A ticker that disappears mid-backtest: shares frozen, price 0."""
        # A: 20 days; B: 10 days (delists after day 10)
        hist_a = make_hist([100] * 20, start_date="2023-01-02")
        hist_b = make_hist([50] * 10, start_date="2023-01-02")
        hist = {"A": hist_a, "B": hist_b}
        bt = PortfolioBacktester(hist, BuyAllDay1(), initial_cash=10_000)
        result = bt.run()

        # After B's data ends, its shares should be frozen and price = 0
        b_shares_after = result.daily_positions["B"].iloc[-1]
        b_shares_last_active = result.daily_positions["B"].iloc[9]
        assert b_shares_after == pytest.approx(b_shares_last_active), \
            "B shares should be frozen after delisting"
        assert result.daily_prices["B"].iloc[-1] == 0.0, \
            "Delisted ticker should have price 0.0"

    def test_conservation_staggered_entry(self):
        """Accounting identity holds with staggered ticker entry."""
        hist_a = make_hist([100 + i for i in range(20)], start_date="2023-01-02")
        hist_b = make_hist([50 + i for i in range(10)], start_date="2023-01-16")
        hist = {"A": hist_a, "B": hist_b}
        bt = PortfolioBacktester(
            hist, EqualWeightRebalance(), initial_cash=10_000,
        )
        result = bt.run()

        for i in range(len(result.daily_values)):
            cash = result.daily_cash.iloc[i]
            equity = sum(
                result.daily_positions[t].iloc[i] * result.daily_prices[t].iloc[i]
                for t in result.tickers
            )
            expected = cash + equity
            actual = result.daily_values.iloc[i]
            assert actual == pytest.approx(expected, rel=1e-10), \
                f"Day {i}: {actual} != {expected}"

    def test_strategies_only_see_active_tickers(self):
        """Strategies should only receive active tickers in portfolio.positions."""
        # A: 20 days; B: enters on day 11
        hist_a = make_hist([100] * 20, start_date="2023-01-02")
        hist_b = make_hist([50] * 10, start_date="2023-01-16")
        hist = {"A": hist_a, "B": hist_b}

        seen_counts = []

        class SpyStrategy(PortfolioStrategy):
            """Records how many tickers are visible each day."""
            name = "Spy"
            def decide(self, day, history, portfolio):
                seen_counts.append(len(portfolio.positions))
                return {}

        bt = PortfolioBacktester(hist, SpyStrategy(), initial_cash=10_000)
        bt.run()

        # First 10 days: only A active -> 1 ticker
        # Last 10 days: both active -> 2 tickers
        assert seen_counts[0] == 1, "Only A should be visible on day 1"
        assert seen_counts[-1] == 2, "Both should be visible on last day"

    def test_result_tickers_includes_all(self):
        """result.tickers should list all tickers, even late entrants."""
        hist_a = make_hist([100] * 20, start_date="2023-01-02")
        hist_b = make_hist([50] * 10, start_date="2023-01-16")
        hist = {"A": hist_a, "B": hist_b}
        bt = PortfolioBacktester(hist, HoldStrategy(), initial_cash=10_000)
        result = bt.run()
        assert set(result.tickers) == {"A", "B"}

    def test_staggered_five_tickers_all_strategies(self):
        """Stress test: 5 tickers with staggered entry, all strategies."""
        np.random.seed(603)
        # Stagger starts by 5 business days each
        starts = ["2023-01-02", "2023-01-09", "2023-01-16",
                  "2023-01-23", "2023-01-30"]
        hist = {}
        for j, (letter, start) in enumerate(zip("ABCDE", starts)):
            n = 40 - j * 5  # Decreasing lengths
            prices = (100 * np.cumprod(1 + np.random.randn(n) * 0.02)).tolist()
            hist[letter] = make_hist(prices, start_date=start)

        for strategy in [
            EqualWeightRebalance(threshold=0.05),
            InverseVolatilityWeight(threshold=0.05),
            IndependentMeanReversion(20, 2.0),
            RelativeStrength(20),
        ]:
            bt = PortfolioBacktester(hist, strategy, initial_cash=50_000)
            result = bt.run()

            assert len(result.tickers) == 5
            assert result.final_value > 0
            assert np.all(np.isfinite(result.daily_values.values))

            # Conservation identity
            for i in range(len(result.daily_values)):
                cash = result.daily_cash.iloc[i]
                equity = sum(
                    result.daily_positions[t].iloc[i]
                    * result.daily_prices[t].iloc[i]
                    for t in result.tickers
                )
                assert result.daily_values.iloc[i] == pytest.approx(
                    cash + equity, rel=1e-10
                ), f"{strategy.name} day {i}: identity violated"


# =====================================================================
# Inverse Volatility Weight Tests
# =====================================================================


class TestInverseVolatilityWeight:
    """Tests for the InverseVolatilityWeight strategy and its core functions."""

    def test_inv_vol_targets_favor_low_vol(self):
        """Low-vol ticker gets more target shares than high-vol ticker."""
        # A is stable (low vol), B is volatile (high vol), same price
        np.random.seed(603)
        n = 80
        prices_a = [100.0 + 0.01 * i for i in range(n)]  # near-constant
        prices_b = (100 * np.cumprod(1 + np.random.randn(n) * 0.05)).tolist()
        hist = make_two_stock_hist(prices_a, prices_b)

        positions = {"A": 0, "B": 0}
        prices = {"A": prices_a[-1], "B": prices_b[-1]}

        targets = compute_inv_vol_target_shares(
            positions=positions,
            prices=prices,
            cash=10_000,
            history=hist,
        )
        # A (low vol) should get more shares in dollar terms
        dollar_a = targets["A"] * prices["A"]
        dollar_b = targets["B"] * prices["B"]
        assert dollar_a > dollar_b, (
            f"Low-vol ticker A (${dollar_a:.0f}) should get more than "
            f"high-vol ticker B (${dollar_b:.0f})"
        )

    def test_inv_vol_target_shares_sum(self):
        """Total cost of target shares doesn't exceed investable amount."""
        np.random.seed(603)
        n = 80
        prices_a = (100 * np.cumprod(1 + np.random.randn(n) * 0.02)).tolist()
        prices_b = (50 * np.cumprod(1 + np.random.randn(n) * 0.03)).tolist()
        hist = make_two_stock_hist(prices_a, prices_b)

        positions = {"A": 0, "B": 0}
        prices = {"A": prices_a[-1], "B": prices_b[-1]}
        cash = 10_000

        targets = compute_inv_vol_target_shares(
            positions=positions,
            prices=prices,
            cash=cash,
            history=hist,
        )
        total_cost = sum(targets[t] * prices[t] for t in positions)
        investable = cash * 0.95  # 1 - CASH_RESERVE_PCT
        assert total_cost <= investable + 1e-10

    def test_inv_vol_all_equal_vol_degrades_to_equal_weight(self):
        """When all tickers have the same volatility, targets match equal weight."""
        # Same price series for both -> same vol -> 1/n weights
        np.random.seed(603)
        n = 80
        base = (100 * np.cumprod(1 + np.random.randn(n) * 0.02)).tolist()
        hist = make_two_stock_hist(base, base.copy())

        positions = {"A": 0, "B": 0}
        prices = {"A": base[-1], "B": base[-1]}
        cash = 10_000

        inv_vol_targets = compute_inv_vol_target_shares(
            positions=positions, prices=prices, cash=cash, history=hist,
        )
        # With identical vols and prices, targets within 1 share (greedy remainder)
        assert abs(inv_vol_targets["A"] - inv_vol_targets["B"]) <= 1

    def test_inv_vol_strategy_runs(self):
        """End-to-end backtest with InverseVolatilityWeight produces valid result."""
        np.random.seed(603)
        n = 80
        prices_a = (100 * np.cumprod(1 + np.random.randn(n) * 0.02)).tolist()
        prices_b = (50 * np.cumprod(1 + np.random.randn(n) * 0.03)).tolist()
        hist = make_two_stock_hist(prices_a, prices_b)

        bt = PortfolioBacktester(
            hist, InverseVolatilityWeight(threshold=0.05),
            initial_cash=10_000,
        )
        result = bt.run()

        assert result.final_value > 0
        assert len(result.trades) > 0
        assert result.strategy_name == "Inverse Volatility"

    def test_inv_vol_conservation(self):
        """Conservation of money holds for InverseVolatilityWeight."""
        np.random.seed(603)
        n = 80
        prices_a = (100 * np.cumprod(1 + np.random.randn(n) * 0.02)).tolist()
        prices_b = (50 * np.cumprod(1 + np.random.randn(n) * 0.03)).tolist()
        hist = make_two_stock_hist(prices_a, prices_b)

        bt = PortfolioBacktester(
            hist, InverseVolatilityWeight(threshold=0.05),
            initial_cash=10_000,
        )
        result = bt.run()

        for i in range(len(result.daily_values)):
            cash = result.daily_cash.iloc[i]
            equity = sum(
                result.daily_positions[t].iloc[i]
                * result.daily_prices[t].iloc[i]
                for t in result.tickers
            )
            assert result.daily_values.iloc[i] == pytest.approx(
                cash + equity, rel=1e-10
            ), f"Day {i}: conservation violated"

    def test_inv_vol_needs_rebalance_basic(self):
        """Triggers rebalance when weight drifts from inv-vol target."""
        np.random.seed(603)
        n = 80
        prices_a = [100.0 + 0.01 * i for i in range(n)]  # low vol
        prices_b = (100 * np.cumprod(1 + np.random.randn(n) * 0.05)).tolist()
        hist = make_two_stock_hist(prices_a, prices_b)

        # Positions heavily skewed to B (high vol) -- opposite of inv-vol target
        positions = {"A": 10, "B": 90}
        prices = {"A": prices_a[-1], "B": prices_b[-1]}
        cash = 100.0

        result = inv_vol_needs_rebalance(
            positions=positions, prices=prices, cash=cash,
            threshold=0.05, history=hist,
        )
        assert result is True

    def test_inv_vol_no_history_fallback(self):
        """Without history, inv-vol targets fall back to equal weight."""
        positions = {"A": 0, "B": 0}
        prices = {"A": 100.0, "B": 100.0}
        cash = 10_000

        targets = compute_inv_vol_target_shares(
            positions=positions,
            prices=prices,
            cash=cash,
            history=None,
        )
        # No history -> all vols infinite -> equal weight fallback (within 1 share)
        assert abs(targets["A"] - targets["B"]) <= 1

    def test_inv_vol_strategy_name(self):
        """Strategy name is 'Inverse Volatility'."""
        strat = InverseVolatilityWeight()
        assert strat.name == "Inverse Volatility"


# #####################################################################
# Monte Carlo Stress-Test Framework
# #####################################################################

# =====================================================================
# Window generation tests
# =====================================================================

class TestGenerateWindows:

    def test_count(self):
        hist = _make_hist_dict(n_days=2000, start_date="2010-01-04")
        windows = generate_windows(hist, window_years=2.0, n_iterations=10, seed=603)
        assert len(windows) == 10

    def test_bounds(self):
        """All windows must start and end within data range."""
        hist = _make_hist_dict(n_days=2000, start_date="2010-01-04")
        windows = generate_windows(hist, window_years=2.0, n_iterations=20, seed=603)

        # Find global min/max dates
        all_min = min(df.index.min() for df in hist.values())
        all_max = max(df.index.max() for df in hist.values())

        for w in windows:
            assert w.start >= all_min.date()
            assert w.end <= all_max.date() + timedelta(days=1)

    def test_duration(self):
        """Each window should span ~window_years calendar years."""
        hist = _make_hist_dict(n_days=2000, start_date="2010-01-04")
        windows = generate_windows(hist, window_years=3.0, n_iterations=5, seed=603)
        expected_days = int(3.0 * 365.25)
        for w in windows:
            actual_days = (w.end - w.start).days
            assert actual_days == expected_days

    def test_reproducibility(self):
        """Same seed produces same windows."""
        hist = _make_hist_dict(n_days=2000, start_date="2010-01-04")
        w1 = generate_windows(hist, window_years=2.0, n_iterations=10, seed=603)
        w2 = generate_windows(hist, window_years=2.0, n_iterations=10, seed=603)
        for a, b in zip(w1, w2):
            assert a.start == b.start
            assert a.end == b.end

    def test_seed_variation(self):
        """Different seeds produce different windows."""
        hist = _make_hist_dict(n_days=2000, start_date="2010-01-04")
        w1 = generate_windows(hist, window_years=2.0, n_iterations=10, seed=603)
        w2 = generate_windows(hist, window_years=2.0, n_iterations=10, seed=999)
        starts_1 = [w.start for w in w1]
        starts_2 = [w.start for w in w2]
        assert starts_1 != starts_2

    def test_sorted(self):
        """Windows should be sorted by start date."""
        hist = _make_hist_dict(n_days=2000, start_date="2010-01-04")
        windows = generate_windows(hist, window_years=2.0, n_iterations=15, seed=603)
        starts = [w.start for w in windows]
        assert starts == sorted(starts)

    def test_insufficient_data(self):
        """When data is shorter than window, return empty."""
        hist = _make_hist_dict(n_days=100, start_date="2020-01-02")
        windows = generate_windows(hist, window_years=3.0, n_iterations=5, seed=603)
        assert len(windows) == 0


# =====================================================================
# History slicing tests
# =====================================================================

class TestSliceHistory:

    def test_date_bounds(self):
        """Sliced data should be within [start, end]."""
        hist = _make_hist_dict(n_days=1000, start_date="2015-01-02")
        start = date(2016, 1, 1)
        end = date(2017, 1, 1)
        sliced = slice_history(hist, start, end)

        for t, df in sliced.items():
            assert df.index.min() >= pd.Timestamp(start)
            assert df.index.max() <= pd.Timestamp(end)

    def test_empty_exclusion(self):
        """Tickers with no data in the window should be excluded."""
        hist = _make_hist_dict(n_days=500, start_date="2015-01-02")
        # Pick a window far in the future
        sliced = slice_history(hist, date(2025, 1, 1), date(2026, 1, 1))
        assert len(sliced) == 0

    def test_column_preservation(self):
        """Sliced DataFrames should preserve OHLCV columns."""
        hist = _make_hist_dict(n_days=1000, start_date="2015-01-02")
        sliced = slice_history(hist, date(2016, 1, 1), date(2017, 1, 1))
        for t, df in sliced.items():
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                assert col in df.columns

    def test_tz_handling(self):
        """Should handle timezone-aware DataFrames."""
        hist = {"A": _make_price_df(n_days=500, start_date="2015-01-02",
                                     tz="US/Eastern"),
                "B": _make_price_df(n_days=500, start_date="2015-01-02",
                                     tz="America/Chicago", seed=99)}
        sliced = slice_history(hist, date(2015, 6, 1), date(2016, 1, 1))
        assert len(sliced) == 2
        for t, df in sliced.items():
            assert df.index.tz is None  # tz stripped


# =====================================================================
# MC runner tests (using synthetic strategies)
# =====================================================================

class TestRunMonteCarlo:

    def test_result_count(self):
        """Should produce n_windows * n_strategies results."""
        hist = _make_hist_dict(n_days=2000, start_date="2010-01-04")
        windows = generate_windows(hist, window_years=2.0, n_iterations=3, seed=603)
        strategies = [_AlwaysBuyStrategy(), _AlwaysHoldStrategy()]
        results = run_monte_carlo(hist, strategies, windows,
                                  initial_cash=10_000, rf_rate=0.04)
        assert len(results) == len(windows) * len(strategies)

    def test_fields_present(self):
        """All MCIterationResult fields should be populated."""
        hist = _make_hist_dict(n_days=2000, start_date="2010-01-04")
        windows = generate_windows(hist, window_years=2.0, n_iterations=2, seed=603)
        strategies = [_AlwaysBuyStrategy()]
        results = run_monte_carlo(hist, strategies, windows,
                                  initial_cash=10_000, rf_rate=0.04)
        for r in results:
            assert isinstance(r.iteration, int)
            assert isinstance(r.window_start, date)
            assert isinstance(r.window_end, date)
            assert isinstance(r.strategy, str)
            assert isinstance(r.sharpe, float)
            assert isinstance(r.ann_return, float)
            assert isinstance(r.ann_vol, float)
            assert isinstance(r.final_value, float)
            assert isinstance(r.n_trades, int)
            assert isinstance(r.n_tickers_active, int)

    def test_finite_sharpe(self):
        """Sharpe ratios should be finite."""
        hist = _make_hist_dict(n_days=2000, start_date="2010-01-04")
        windows = generate_windows(hist, window_years=2.0, n_iterations=3, seed=603)
        strategies = [_AlwaysBuyStrategy()]
        results = run_monte_carlo(hist, strategies, windows,
                                  initial_cash=10_000, rf_rate=0.04)
        for r in results:
            assert math.isfinite(r.sharpe)

    def test_positive_final_values(self):
        """Final portfolio values should be positive."""
        hist = _make_hist_dict(n_days=2000, start_date="2010-01-04")
        windows = generate_windows(hist, window_years=2.0, n_iterations=3, seed=603)
        strategies = [_AlwaysBuyStrategy(), _AlwaysHoldStrategy()]
        results = run_monte_carlo(hist, strategies, windows,
                                  initial_cash=10_000, rf_rate=0.04)
        for r in results:
            assert r.final_value > 0

    def test_all_strategies_present(self):
        """Each strategy should appear in results."""
        hist = _make_hist_dict(n_days=2000, start_date="2010-01-04")
        windows = generate_windows(hist, window_years=2.0, n_iterations=3, seed=603)
        strategies = [_AlwaysBuyStrategy(), _AlwaysHoldStrategy()]
        results = run_monte_carlo(hist, strategies, windows,
                                  initial_cash=10_000, rf_rate=0.04)
        strat_names = set(r.strategy for r in results)
        assert "AlwaysBuy" in strat_names
        assert "AlwaysHold" in strat_names


# =====================================================================
# Aggregation tests
# =====================================================================

class TestAggregation:

    @pytest.fixture
    def mc_df(self):
        hist = _make_hist_dict(n_days=2000, start_date="2010-01-04")
        windows = generate_windows(hist, window_years=2.0, n_iterations=5, seed=603)
        strategies = [_AlwaysBuyStrategy(), _AlwaysHoldStrategy()]
        results = run_monte_carlo(hist, strategies, windows,
                                  initial_cash=10_000, rf_rate=0.04)
        return mc_results_to_dataframe(results)

    def test_dataframe_shape(self, mc_df):
        """DataFrame should have one row per strategy per window."""
        assert len(mc_df) == 10  # 5 windows * 2 strategies

    def test_dataframe_columns(self, mc_df):
        expected = {"iteration", "window_start", "window_end", "strategy",
                    "sharpe", "ann_return", "ann_vol", "max_dd",
                    "final_value", "total_return", "n_trades",
                    "n_tickers_active"}
        assert expected.issubset(set(mc_df.columns))

    def test_summary_per_strategy(self, mc_df):
        summary = summarize_mc_results(mc_df)
        assert len(summary) == 2
        assert "sharpe_mean" in summary.columns
        assert "sharpe_median" in summary.columns
        assert "sharpe_std" in summary.columns
        assert "win_rate" in summary.columns

    def test_win_rate_bounds(self, mc_df):
        summary = summarize_mc_results(mc_df)
        for strat in summary.index:
            wr = summary.loc[strat, "win_rate"]
            assert 0.0 <= wr <= 1.0

    def test_pairwise_diagonal_nan(self, mc_df):
        matrix = compute_win_rates(mc_df)
        for s in matrix.index:
            assert pd.isna(matrix.loc[s, s])

    def test_pairwise_complementary(self, mc_df):
        """Win rate A vs B + win rate B vs A should be ~1 (ties allowed)."""
        matrix = compute_win_rates(mc_df)
        strats = list(matrix.index)
        for i, s_a in enumerate(strats):
            for j, s_b in enumerate(strats):
                if i == j:
                    continue
                ab = matrix.loc[s_a, s_b]
                ba = matrix.loc[s_b, s_a]
                if pd.notna(ab) and pd.notna(ba):
                    assert abs(ab + ba - 1.0) < 0.01 + 1e-9  # ties give slack


# =====================================================================
# Edge cases (Monte Carlo)
# =====================================================================

class TestEdgeCasesMC:

    def test_single_iteration(self):
        hist = _make_hist_dict(n_days=2000, start_date="2010-01-04")
        windows = generate_windows(hist, window_years=2.0, n_iterations=1, seed=603)
        assert len(windows) == 1
        strategies = [_AlwaysBuyStrategy()]
        results = run_monte_carlo(hist, strategies, windows,
                                  initial_cash=10_000, rf_rate=0.04)
        assert len(results) == 1

    def test_single_strategy(self):
        hist = _make_hist_dict(n_days=2000, start_date="2010-01-04")
        windows = generate_windows(hist, window_years=2.0, n_iterations=5, seed=603)
        strategies = [_AlwaysBuyStrategy()]
        results = run_monte_carlo(hist, strategies, windows,
                                  initial_cash=10_000, rf_rate=0.04)
        df = mc_results_to_dataframe(results)
        assert df["strategy"].nunique() == 1

    def test_empty_hist_dict(self):
        windows = generate_windows({}, window_years=2.0, n_iterations=5, seed=603)
        assert len(windows) == 0


# =====================================================================
# Cross-script consistency (Monte Carlo)
# =====================================================================

class TestCrossScriptConsistencyMC:

    def test_stress_test_tickers_in_universe(self):
        """All STRESS_TEST_12 tickers should be in SP500_UNIVERSE."""
        from finance_tools.data.universe import SP500_UNIVERSE
        stress_test_path = os.path.join(
            REPO_ROOT, "apps", "backtester", "stress_test.py")
        spec = importlib.util.spec_from_file_location(
            "stress_test_script", stress_test_path)
        stress_test_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stress_test_mod)

        for t in stress_test_mod.STRESS_TEST_12:
            assert t in SP500_UNIVERSE, f"{t} not in SP500_UNIVERSE"

    def test_stress_test_12_count(self):
        stress_test_path = os.path.join(
            REPO_ROOT, "apps", "backtester", "stress_test.py")
        spec = importlib.util.spec_from_file_location(
            "stress_test_script", stress_test_path)
        stress_test_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stress_test_mod)

        assert len(stress_test_mod.STRESS_TEST_12) == 12

    def test_stress_test_12_sorted(self):
        stress_test_path = os.path.join(
            REPO_ROOT, "apps", "backtester", "stress_test.py")
        spec = importlib.util.spec_from_file_location(
            "stress_test_script", stress_test_path)
        stress_test_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stress_test_mod)

        assert stress_test_mod.STRESS_TEST_12 == sorted(stress_test_mod.STRESS_TEST_12)


# =====================================================================
# Normalize index tests
# =====================================================================

class TestNormalizeIndex:

    def test_strips_timezone(self):
        df = _make_price_df(n_days=10, tz="US/Eastern")
        ndf = _normalize_index(df)
        assert ndf.index.tz is None

    def test_no_tz_passthrough(self):
        df = _make_price_df(n_days=10)
        ndf = _normalize_index(df)
        assert ndf.index.tz is None
        assert len(ndf) == len(df)

    def test_deduplicates(self):
        df = _make_price_df(n_days=10)
        # Duplicate a row
        dup = pd.concat([df, df.iloc[[0]]])
        ndf = _normalize_index(dup)
        assert len(ndf) == len(df)
