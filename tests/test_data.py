"""
Tests for Finance --- Universe (universe.py) and Results Store (results.py).

Covers:
- Universe: ticker count, sector count, derived constants, helpers
- ResultsStore: save/load round-trip, queries, compare, delete
- Cross-script: regime uses REGIME_25, app uses TRADING_ASSISTANT_10
"""

import importlib.util
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from finance_tools.data.universe import (
    SP500_UNIVERSE, ALL_TICKERS, SECTORS,
    REGIME_25, TRADING_ASSISTANT_10,
    tickers_by_sector, get_sector, validate_tickers,
)
from finance_tools.data.results import ResultsStore
from finance_tools.backtest.portfolio import (
    PortfolioBacktester, PortfolioBacktestResult, PortfolioState,
)
from finance_tools.backtest.engine import Action, ActionType
from finance_tools.strategies.portfolio import EqualWeightRebalance

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =====================================================================
# Helpers
# =====================================================================

def make_hist(prices, start_date="2023-01-02"):
    """Build a minimal OHLCV DataFrame from closing prices."""
    n = len(prices)
    dates = pd.bdate_range(start_date, periods=n)
    return pd.DataFrame({
        "Open": prices,
        "High": [p * 1.01 for p in prices],
        "Low": [p * 0.99 for p in prices],
        "Close": prices,
        "Volume": [1_000_000] * n,
        "Dividends": [0.0] * n,
    }, index=dates)


def make_portfolio_result(tickers=None, n_days=100, strategy_name="Test EW"):
    """Create a minimal PortfolioBacktestResult for testing."""
    if tickers is None:
        tickers = ["AAPL", "MSFT", "NVDA"]

    hist_dict = {}
    for t in tickers:
        np.random.seed(hash(t) % 2**31)
        prices = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
        prices = np.maximum(prices, 1.0)
        hist_dict[t] = make_hist(prices.tolist())

    strat = EqualWeightRebalance(threshold=0.05)
    bt = PortfolioBacktester(hist_dict, strat, initial_cash=10_000)
    res = bt.run()
    res.rf_rate = 0.04
    return res


# =====================================================================
# Universe Tests
# =====================================================================

class TestUniverseTickers:

    def test_total_count(self):
        assert len(SP500_UNIVERSE) == 100

    def test_all_tickers_sorted(self):
        assert ALL_TICKERS == sorted(ALL_TICKERS)

    def test_all_tickers_count(self):
        assert len(ALL_TICKERS) == 100

    def test_all_tickers_match_universe(self):
        assert set(ALL_TICKERS) == set(SP500_UNIVERSE.keys())

    def test_no_duplicates(self):
        assert len(ALL_TICKERS) == len(set(ALL_TICKERS))

    def test_all_strings(self):
        for t in ALL_TICKERS:
            assert isinstance(t, str)
            assert len(t) > 0
        for t, s in SP500_UNIVERSE.items():
            assert isinstance(s, str)
            assert len(s) > 0


class TestUniverseSectors:

    def test_eleven_sectors(self):
        assert len(SECTORS) == 11

    def test_expected_sectors(self):
        expected = {
            "Technology", "Financials", "Healthcare",
            "Consumer Discretionary", "Communication", "Consumer Staples",
            "Industrials", "Energy", "Utilities", "Materials", "Real Estate",
        }
        assert set(SECTORS.keys()) == expected

    def test_sector_counts(self):
        expected = {
            "Technology": 22,
            "Financials": 10,
            "Healthcare": 10,
            "Consumer Discretionary": 10,
            "Communication": 6,
            "Consumer Staples": 8,
            "Industrials": 9,
            "Energy": 8,
            "Utilities": 4,
            "Materials": 8,
            "Real Estate": 5,
        }
        for sector, count in expected.items():
            assert len(SECTORS[sector]) == count, f"{sector}: expected {count}"

    def test_sectors_cover_all_tickers(self):
        all_from_sectors = set()
        for tickers in SECTORS.values():
            all_from_sectors.update(tickers)
        assert all_from_sectors == set(ALL_TICKERS)

    def test_sectors_sorted(self):
        for sector, tickers in SECTORS.items():
            assert tickers == sorted(tickers), f"{sector} not sorted"


class TestUniverseSubsets:

    def test_regime_25_count(self):
        assert len(REGIME_25) == 25

    def test_regime_25_subset_of_universe(self):
        assert set(REGIME_25).issubset(set(ALL_TICKERS))

    def test_regime_25_sorted(self):
        assert REGIME_25 == sorted(REGIME_25)

    def test_regime_25_no_duplicates(self):
        assert len(REGIME_25) == len(set(REGIME_25))

    def test_trading_assistant_10_count(self):
        assert len(TRADING_ASSISTANT_10) == 10

    def test_trading_assistant_10_subset_of_universe(self):
        assert set(TRADING_ASSISTANT_10).issubset(set(ALL_TICKERS))

    def test_trading_assistant_10_sorted(self):
        assert TRADING_ASSISTANT_10 == sorted(TRADING_ASSISTANT_10)

    def test_trading_assistant_10_no_duplicates(self):
        assert len(TRADING_ASSISTANT_10) == len(set(TRADING_ASSISTANT_10))


class TestUniverseHelpers:

    def test_tickers_by_sector_known(self):
        tech = tickers_by_sector("Technology")
        assert "AAPL" in tech
        assert "MSFT" in tech
        assert len(tech) == 22

    def test_tickers_by_sector_unknown_raises(self):
        with pytest.raises(KeyError):
            tickers_by_sector("Nonexistent")

    def test_get_sector_known(self):
        assert get_sector("AAPL") == "Technology"
        assert get_sector("JPM") == "Financials"
        assert get_sector("F") == "Consumer Discretionary"

    def test_get_sector_unknown(self):
        assert get_sector("ZZZZZ") is None

    def test_validate_tickers_all_valid(self):
        result = validate_tickers(["AAPL", "MSFT", "NVDA"])
        assert result == ["AAPL", "MSFT", "NVDA"]

    def test_validate_tickers_filters_invalid(self):
        result = validate_tickers(["AAPL", "FAKE", "MSFT", "BOGUS"])
        assert result == ["AAPL", "MSFT"]

    def test_validate_tickers_preserves_order(self):
        result = validate_tickers(["NVDA", "AAPL", "MSFT"])
        assert result == ["NVDA", "AAPL", "MSFT"]

    def test_validate_tickers_empty(self):
        assert validate_tickers([]) == []
        assert validate_tickers(["FAKE1", "FAKE2"]) == []


# =====================================================================
# ResultsStore Tests
# =====================================================================

@pytest.fixture
def temp_store():
    """Create a ResultsStore backed by a temp file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = ResultsStore(db_path=path)
    yield store
    store.close()
    os.unlink(path)


class TestResultsStoreSave:

    def test_save_returns_run_id(self, temp_store):
        result = make_portfolio_result()
        run_id = temp_store.save(result)
        assert isinstance(run_id, int)
        assert run_id >= 1

    def test_save_increments_id(self, temp_store):
        result = make_portfolio_result()
        id1 = temp_store.save(result)
        id2 = temp_store.save(result)
        assert id2 == id1 + 1

    def test_save_preserves_notes(self, temp_store):
        result = make_portfolio_result()
        run_id = temp_store.save(result, notes="test note 123")
        run = temp_store.get_run(run_id)
        assert run["notes"] == "test note 123"

    def test_save_preserves_runner_script(self, temp_store):
        result = make_portfolio_result()
        run_id = temp_store.save(result, runner_script="test_runner.py")
        run = temp_store.get_run(run_id)
        assert run["runner_script"] == "test_runner.py"

    def test_save_stores_strategy_name(self, temp_store):
        result = make_portfolio_result()
        run_id = temp_store.save(result)
        run = temp_store.get_run(run_id)
        assert run["strategy_name"] == result.strategy_name

    def test_save_stores_metrics(self, temp_store):
        result = make_portfolio_result()
        run_id = temp_store.save(result)
        run = temp_store.get_run(run_id)
        assert abs(run["final_value"] - result.final_value) < 0.01
        assert abs(run["total_return"] - result.total_return) < 1e-6
        assert abs(run["sharpe_ratio"] - result.sharpe_ratio) < 1e-6
        assert run["n_trading_days"] == len(result.daily_values)
        assert run["n_tickers"] == len(result.tickers)

    def test_save_stores_tickers_as_json(self, temp_store):
        result = make_portfolio_result(tickers=["AAPL", "MSFT"])
        run_id = temp_store.save(result)
        run = temp_store.get_run(run_id)
        tickers = json.loads(run["tickers"])
        assert set(tickers) == {"AAPL", "MSFT"}


class TestResultsStoreTickers:

    def test_ticker_contributions_saved(self, temp_store):
        tickers = ["AAPL", "MSFT", "NVDA"]
        result = make_portfolio_result(tickers=tickers)
        run_id = temp_store.save(result)
        contribs = temp_store.get_ticker_contributions(run_id)
        assert len(contribs) == 3
        saved_tickers = {c["ticker"] for c in contribs}
        assert saved_tickers == set(tickers)

    def test_ticker_contributions_have_sector(self, temp_store):
        result = make_portfolio_result(tickers=["AAPL", "JPM"])
        run_id = temp_store.save(result)
        contribs = temp_store.get_ticker_contributions(run_id)
        sectors = {c["ticker"]: c["sector"] for c in contribs}
        assert sectors["AAPL"] == "Technology"
        assert sectors["JPM"] == "Financials"

    def test_ticker_contributions_empty_for_nonexistent_run(self, temp_store):
        contribs = temp_store.get_ticker_contributions(9999)
        assert contribs == []


class TestResultsStoreQuery:

    def test_list_runs_empty(self, temp_store):
        runs = temp_store.list_runs()
        assert runs == []

    def test_list_runs_returns_saved(self, temp_store):
        result = make_portfolio_result()
        temp_store.save(result)
        temp_store.save(result, notes="second")
        runs = temp_store.list_runs()
        assert len(runs) == 2

    def test_list_runs_limit(self, temp_store):
        result = make_portfolio_result()
        for _ in range(5):
            temp_store.save(result)
        runs = temp_store.list_runs(limit=3)
        assert len(runs) == 3

    def test_list_runs_filter_by_strategy(self, temp_store):
        res1 = make_portfolio_result()
        temp_store.save(res1)
        # Save with a known strategy name
        run = temp_store.get_run(1)
        strat_name = run["strategy_name"]
        runs = temp_store.list_runs(strategy=strat_name)
        assert len(runs) == 1
        runs_none = temp_store.list_runs(strategy="Nonexistent Strategy")
        assert len(runs_none) == 0

    def test_get_run_nonexistent(self, temp_store):
        assert temp_store.get_run(9999) is None

    def test_compare_formats_table(self, temp_store):
        result = make_portfolio_result()
        id1 = temp_store.save(result)
        id2 = temp_store.save(result, notes="second")
        table = temp_store.compare([id1, id2])
        assert "Metric" in table
        assert "Strategy" in table
        assert "Sharpe" in table

    def test_compare_no_runs(self, temp_store):
        table = temp_store.compare([999, 1000])
        assert "No matching" in table

    def test_best_by_sharpe(self, temp_store):
        result = make_portfolio_result()
        for _ in range(3):
            temp_store.save(result)
        best = temp_store.best_by("sharpe_ratio", n=2)
        assert len(best) == 2

    def test_best_by_max_drawdown(self, temp_store):
        result = make_portfolio_result()
        for _ in range(3):
            temp_store.save(result)
        best = temp_store.best_by("max_drawdown", n=2)
        assert len(best) == 2


class TestResultsStoreDelete:

    def test_delete_existing(self, temp_store):
        result = make_portfolio_result()
        run_id = temp_store.save(result)
        assert temp_store.delete(run_id) is True
        assert temp_store.get_run(run_id) is None

    def test_delete_removes_ticker_contributions(self, temp_store):
        result = make_portfolio_result()
        run_id = temp_store.save(result)
        assert len(temp_store.get_ticker_contributions(run_id)) > 0
        temp_store.delete(run_id)
        assert len(temp_store.get_ticker_contributions(run_id)) == 0

    def test_delete_nonexistent(self, temp_store):
        assert temp_store.delete(9999) is False


class TestResultsStoreSingleStock:

    def test_save_single(self, temp_store):
        from finance_tools.backtest.engine import BacktestResult
        prices = [100 + i * 0.5 for i in range(50)]
        dates = pd.bdate_range("2023-01-02", periods=50)
        run_id = temp_store.save_single(BacktestResult(
            strategy_name="Buy and Hold",
            initial_cash=10_000.0,
            final_value=10_500.0,
            trades=[],
            daily_values=pd.Series(prices, index=dates),
            daily_cash=pd.Series([500.0] * 50, index=dates),
            daily_shares=pd.Series([95.0] * 50, index=dates),
            rf_rate=0.04,
        ))
        assert run_id >= 1
        run = temp_store.get_run(run_id)
        assert run["strategy_name"] == "Buy and Hold"
        assert run["n_tickers"] == 1


class TestResultsStoreDailyValues:

    def test_save_stores_daily_values(self, temp_store):
        result = make_portfolio_result(n_days=100)
        run_id = temp_store.save(result)
        rows = temp_store.get_daily_values(run_id)
        assert len(rows) == len(result.daily_values)

    def test_get_daily_values_roundtrip(self, temp_store):
        result = make_portfolio_result(n_days=50)
        run_id = temp_store.save(result)
        rows = temp_store.get_daily_values(run_id)
        # First row matches
        first_date = str(result.daily_values.index[0].date())
        assert rows[0]["date"] == first_date
        assert abs(rows[0]["portfolio_value"] - float(result.daily_values.iloc[0])) < 0.01
        assert abs(rows[0]["cash_value"] - float(result.daily_cash.iloc[0])) < 0.01
        # Last row matches
        last_date = str(result.daily_values.index[-1].date())
        assert rows[-1]["date"] == last_date
        assert abs(rows[-1]["portfolio_value"] - float(result.daily_values.iloc[-1])) < 0.01

    def test_get_daily_values_empty_for_nonexistent_run(self, temp_store):
        rows = temp_store.get_daily_values(9999)
        assert rows == []

    def test_has_daily_values_true_after_save(self, temp_store):
        result = make_portfolio_result()
        run_id = temp_store.save(result)
        assert temp_store.has_daily_values(run_id) is True

    def test_has_daily_values_false_for_nonexistent(self, temp_store):
        assert temp_store.has_daily_values(9999) is False

    def test_get_daily_values_multi_returns_dict(self, temp_store):
        result = make_portfolio_result(n_days=30)
        id1 = temp_store.save(result)
        id2 = temp_store.save(result)
        multi = temp_store.get_daily_values_multi([id1, id2])
        assert isinstance(multi, dict)
        assert set(multi.keys()) == {id1, id2}
        assert len(multi[id1]) == 30
        assert len(multi[id2]) == 30

    def test_delete_removes_daily_values(self, temp_store):
        result = make_portfolio_result()
        run_id = temp_store.save(result)
        assert temp_store.has_daily_values(run_id) is True
        temp_store.delete(run_id)
        assert temp_store.has_daily_values(run_id) is False

    def test_save_single_stores_daily_values(self, temp_store):
        from finance_tools.backtest.engine import BacktestResult
        prices = [100 + i * 0.5 for i in range(50)]
        dates = pd.bdate_range("2023-01-02", periods=50)
        run_id = temp_store.save_single(BacktestResult(
            strategy_name="Buy and Hold",
            initial_cash=10_000.0,
            final_value=10_500.0,
            trades=[],
            daily_values=pd.Series(prices, index=dates),
            daily_cash=pd.Series([500.0] * 50, index=dates),
            daily_shares=pd.Series([95.0] * 50, index=dates),
            rf_rate=0.04,
        ))
        assert temp_store.has_daily_values(run_id) is True
        rows = temp_store.get_daily_values(run_id)
        assert len(rows) == 50


class TestResultsStoreContextManager:

    def test_context_manager(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        with ResultsStore(db_path=path) as store:
            result = make_portfolio_result()
            run_id = store.save(result)
            assert run_id >= 1
        os.unlink(path)


# =====================================================================
# Cross-Script Consistency
# =====================================================================

class TestCrossScriptConsistency:

    def test_regime_adaptive_uses_regime_25(self):
        _regime_path = os.path.join(_REPO_ROOT, "apps", "backtester", "regime.py")
        spec = importlib.util.spec_from_file_location(
            "regime_adaptive_data_test", _regime_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert sorted(mod.TICKERS) == sorted(REGIME_25)

    def test_regime_adaptive_ticker_count(self):
        _regime_path = os.path.join(_REPO_ROOT, "apps", "backtester", "regime.py")
        spec = importlib.util.spec_from_file_location(
            "regime_adaptive_data_test2", _regime_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert len(mod.TICKERS) == 25

    def test_app_default_tickers_match_trading_assistant_10(self):
        _app_path = os.path.join(_REPO_ROOT, "apps", "portfolio_trader", "app.py")
        spec = importlib.util.spec_from_file_location(
            "trading_assistant_app_universe_test", _app_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert sorted(mod.DEFAULT_TICKERS) == sorted(TRADING_ASSISTANT_10)

    def test_app_default_tickers_count(self):
        _app_path = os.path.join(_REPO_ROOT, "apps", "portfolio_trader", "app.py")
        spec = importlib.util.spec_from_file_location(
            "trading_assistant_app_universe_test2", _app_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert len(mod.DEFAULT_TICKERS) == 10

    def test_run_portfolio_backtest_uses_all_tickers(self):
        """run_portfolio_backtest default should be ALL_TICKERS."""
        _run_path = os.path.join(_REPO_ROOT, "apps", "backtester", "run.py")
        spec = importlib.util.spec_from_file_location(
            "run_portfolio_backtest_data_test", _run_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--tickers", nargs="+", default=mod.ALL_TICKERS)
        args = parser.parse_args([])
        assert sorted(args.tickers) == sorted(ALL_TICKERS)


# =====================================================================
# File Structure
# =====================================================================

class TestFileStructure:

    def test_universe_py_exists(self):
        path = os.path.join(_REPO_ROOT, "finance_tools", "data", "universe.py")
        assert os.path.isfile(path)

    def test_results_store_py_exists(self):
        path = os.path.join(_REPO_ROOT, "finance_tools", "data", "results.py")
        assert os.path.isfile(path)

    def test_view_results_py_exists(self):
        path = os.path.join(_REPO_ROOT, "finance_tools", "data", "view_results.py")
        assert os.path.isfile(path)

    def test_gitignore_has_db(self):
        gitignore = os.path.join(_REPO_ROOT, ".gitignore")
        with open(gitignore) as f:
            content = f.read()
        assert "backtest_results.db" in content
