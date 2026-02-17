"""
Tests for Finance — Regime-Adaptive Strategy.

Covers: _normalize_index, classify_regime, compute_indicators,
AdaptiveStrategy decision routing, end-to-end backtest, conservation
of money, file structure, cross-script consistency with thresholds.
"""

import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

from finance_tools.backtest.engine import Action, ActionType
from finance_tools.backtest.portfolio import (
    PortfolioState, PortfolioStrategy, PortfolioBacktester,
    PortfolioBacktestResult,
)
from finance_tools.strategies.portfolio import (
    EqualWeightRebalance, IndependentMeanReversion, RelativeStrength,
)

# Load regime module from apps/backtester/regime.py
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REGIME_PATH = os.path.join(_REPO_ROOT, "apps", "backtester", "regime.py")
_spec = importlib.util.spec_from_file_location("regime_adaptive", _REGIME_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_normalize_index = _mod._normalize_index
classify_regime = _mod.classify_regime
compute_indicators = _mod.compute_indicators
AdaptiveStrategy = _mod.AdaptiveStrategy
VIX_HIGH = _mod.VIX_HIGH
VIX_LOW = _mod.VIX_LOW
TICKERS = _mod.TICKERS
REGIME_COLORS = _mod.REGIME_COLORS
REGIME_LABELS = _mod.REGIME_LABELS


# =====================================================================
# Helpers
# =====================================================================

def make_hist(prices, volumes=None, dividends=None,
              start_date="2023-01-02"):
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


def make_indicator_df(vix_values, spy_values, spy_ma200=None,
                      start_date="2023-01-02"):
    """Build a minimal indicators DataFrame for classify_regime."""
    n = len(vix_values)
    dates = pd.bdate_range(start_date, periods=n)

    if spy_ma200 is None:
        # Default: SPY is always above 200-day MA (bull)
        spy_ma200 = [s * 0.95 for s in spy_values]

    spy_bull = [1.0 if s > m else 0.0
                for s, m in zip(spy_values, spy_ma200)]

    return pd.DataFrame({
        "vix": vix_values,
        "spy_close": spy_values,
        "spy_ma200": spy_ma200,
        "spy_bull": spy_bull,
        "dispersion": [0.01] * n,
        "avg_corr": [0.3] * n,
    }, index=dates)


def make_multi_stock_hist(n_tickers=3, n_days=50, base_price=100.0,
                          start_date="2023-01-02"):
    """Build a dict of n_tickers synthetic histories."""
    np.random.seed(603)
    tickers = [chr(ord("A") + i) for i in range(n_tickers)]
    hist_dict = {}
    for t in tickers:
        prices = [base_price]
        for _ in range(n_days - 1):
            ret = np.random.normal(0.0005, 0.015)
            prices.append(prices[-1] * (1 + ret))
        hist_dict[t] = make_hist(prices, start_date=start_date)
    return hist_dict


# =====================================================================
# _normalize_index
# =====================================================================

class TestNormalizeIndex:

    def test_strips_timezone(self):
        dates = pd.date_range("2023-01-02", periods=5, freq="B",
                              tz="US/Eastern")
        df = pd.DataFrame({"Close": range(5)}, index=dates)
        result = _normalize_index(df)
        assert result.index.tz is None

    def test_normalizes_to_midnight(self):
        dates = pd.to_datetime(["2023-01-02 14:30:00",
                                "2023-01-03 09:15:00",
                                "2023-01-04 16:00:00"])
        df = pd.DataFrame({"Close": [1, 2, 3]}, index=dates)
        result = _normalize_index(df)
        for ts in result.index:
            assert ts.hour == 0 and ts.minute == 0

    def test_removes_duplicate_dates(self):
        dates = pd.to_datetime(["2023-01-02", "2023-01-02", "2023-01-03"])
        df = pd.DataFrame({"Close": [10, 20, 30]}, index=dates)
        result = _normalize_index(df)
        assert len(result) == 2
        assert result.iloc[0]["Close"] == 10  # keeps first

    def test_preserves_data(self):
        df = make_hist([100, 110, 120])
        result = _normalize_index(df)
        assert list(result["Close"]) == [100, 110, 120]

    def test_handles_already_clean_index(self):
        df = make_hist([50, 60])
        result = _normalize_index(df)
        assert len(result) == 2


# =====================================================================
# classify_regime
# =====================================================================

class TestClassifyRegime:

    def test_high_vix_triggers_mean_reversion(self):
        indicators = make_indicator_df(
            vix_values=[30, 28, 26],
            spy_values=[400, 400, 400],
        )
        regime = classify_regime(indicators)
        assert all(regime == "mean_reversion")

    def test_low_vix_bull_triggers_relative_strength(self):
        indicators = make_indicator_df(
            vix_values=[15, 18, 12],
            spy_values=[450, 460, 470],
            spy_ma200=[400, 400, 400],  # SPY > 200MA
        )
        regime = classify_regime(indicators)
        assert all(regime == "relative_strength")

    def test_medium_vix_defaults_to_equal_weight(self):
        indicators = make_indicator_df(
            vix_values=[22, 21, 23],
            spy_values=[400, 400, 400],
            spy_ma200=[390, 390, 390],  # SPY > MA but VIX between 20-25
        )
        regime = classify_regime(indicators)
        assert all(regime == "equal_weight")

    def test_low_vix_bear_defaults_to_equal_weight(self):
        """VIX < 20 but SPY below 200MA -> not calm-bull -> equal weight."""
        indicators = make_indicator_df(
            vix_values=[15, 18, 19],
            spy_values=[380, 370, 360],
            spy_ma200=[400, 400, 400],  # SPY < MA (bear)
        )
        regime = classify_regime(indicators)
        assert all(regime == "equal_weight")

    def test_mixed_regimes(self):
        """Different days get different regimes."""
        indicators = make_indicator_df(
            vix_values=[15, 30, 22],
            spy_values=[450, 400, 400],
            spy_ma200=[400, 400, 400],
        )
        regime = classify_regime(indicators)
        assert regime.iloc[0] == "relative_strength"   # VIX=15, SPY>MA
        assert regime.iloc[1] == "mean_reversion"       # VIX=30
        assert regime.iloc[2] == "equal_weight"          # VIX=22, in between

    def test_boundary_vix_high(self):
        """VIX exactly at threshold (25) should NOT trigger mean_reversion."""
        indicators = make_indicator_df(
            vix_values=[25],
            spy_values=[400],
            spy_ma200=[390],
        )
        regime = classify_regime(indicators)
        assert regime.iloc[0] == "equal_weight"  # > 25 required, not >=

    def test_boundary_vix_low(self):
        """VIX exactly at threshold (20) should NOT trigger relative_strength."""
        indicators = make_indicator_df(
            vix_values=[20],
            spy_values=[450],
            spy_ma200=[400],
        )
        regime = classify_regime(indicators)
        assert regime.iloc[0] == "equal_weight"  # < 20 required, not <=

    def test_calm_bull_overrides_stressed(self):
        """If VIX < 20 (calm-bull wins), it shouldn't also be mean_reversion.
        This can't happen in practice (VIX < 20 and > 25 simultaneously),
        but verify classification priority."""
        indicators = make_indicator_df(
            vix_values=[15],
            spy_values=[450],
            spy_ma200=[400],
        )
        regime = classify_regime(indicators)
        assert regime.iloc[0] == "relative_strength"

    def test_returns_series_same_length(self):
        n = 10
        indicators = make_indicator_df(
            vix_values=[22] * n,
            spy_values=[400] * n,
        )
        regime = classify_regime(indicators)
        assert len(regime) == n
        assert isinstance(regime, pd.Series)


# =====================================================================
# compute_indicators
# =====================================================================

class TestComputeIndicators:

    def _make_synthetic_spy_vix(self, n=250):
        """Create synthetic SPY and VIX DataFrames."""
        dates = pd.bdate_range("2020-01-02", periods=n)
        spy = pd.DataFrame({
            "Open": 400 + np.arange(n) * 0.1,
            "High": 401 + np.arange(n) * 0.1,
            "Low": 399 + np.arange(n) * 0.1,
            "Close": 400 + np.arange(n) * 0.1,
            "Volume": [1_000_000] * n,
        }, index=dates)
        vix = pd.DataFrame({
            "Open": [20.0] * n,
            "High": [21.0] * n,
            "Low": [19.0] * n,
            "Close": [20.0] * n,
            "Volume": [100_000] * n,
        }, index=dates)
        return spy, vix

    def test_returns_dataframe(self):
        spy, vix = self._make_synthetic_spy_vix()
        stock_data = {"A": spy.copy(), "B": spy.copy()}
        result = compute_indicators(stock_data, spy, vix)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self):
        spy, vix = self._make_synthetic_spy_vix()
        stock_data = {"A": spy.copy()}
        result = compute_indicators(stock_data, spy, vix)
        expected = {"vix", "spy_close", "spy_ma200", "spy_bull",
                    "dispersion", "avg_corr"}
        assert expected == set(result.columns)

    def test_vix_smoothed(self):
        """VIX output should be smoothed (5-day rolling mean)."""
        spy, vix = self._make_synthetic_spy_vix(n=10)
        # Make VIX jump on day 5
        vix.loc[vix.index[5], "Close"] = 40.0
        stock_data = {"A": spy.copy()}
        result = compute_indicators(stock_data, spy, vix)
        # Smoothed VIX on day 5 should be < 40 (averaged with neighbors)
        assert result["vix"].iloc[5] < 40.0

    def test_spy_bull_binary(self):
        """spy_bull should be 0.0 or 1.0."""
        spy, vix = self._make_synthetic_spy_vix(n=250)
        stock_data = {"A": spy.copy()}
        result = compute_indicators(stock_data, spy, vix)
        bull_vals = result["spy_bull"].dropna().unique()
        assert all(v in (0.0, 1.0) for v in bull_vals)

    def test_spy_ma200_nan_before_200_days(self):
        """200-day MA should be NaN for first 199 days."""
        spy, vix = self._make_synthetic_spy_vix(n=250)
        stock_data = {"A": spy.copy()}
        result = compute_indicators(stock_data, spy, vix)
        assert pd.isna(result["spy_ma200"].iloc[0])
        assert pd.notna(result["spy_ma200"].iloc[199])

    def test_handles_timezone_mismatch(self):
        """SPY and VIX with different timezones should still work."""
        spy, vix = self._make_synthetic_spy_vix(n=10)
        spy.index = spy.index.tz_localize("US/Eastern")
        vix.index = vix.index.tz_localize("America/Chicago")
        stock_data = {"A": make_hist([100] * 10)}
        result = compute_indicators(stock_data, spy, vix)
        assert isinstance(result, pd.DataFrame)
        assert result.index.tz is None


# =====================================================================
# AdaptiveStrategy --- Decision Routing
# =====================================================================

class TestAdaptiveStrategy:

    @pytest.fixture
    def hist_dict(self):
        return make_multi_stock_hist(n_tickers=3, n_days=50)

    def _make_regime_labels(self, dates, regime):
        """Build a regime_labels Series mapping all dates to one regime."""
        return pd.Series(regime, index=dates)

    def test_defaults_to_equal_weight(self, hist_dict):
        """Unknown dates default to equal_weight."""
        strat = AdaptiveStrategy(regime_labels=pd.Series(dtype=str))
        bt = PortfolioBacktester(hist_dict, strat, initial_cash=10_000)
        res = bt.run()
        # Should complete without error and have trades
        assert res.final_value > 0
        assert res.n_trades > 0

    def test_all_mean_reversion(self, hist_dict):
        """If every day is mean_reversion, should use MR strategy."""
        bt_temp = PortfolioBacktester(hist_dict, EqualWeightRebalance(),
                                      initial_cash=10_000)
        dates = bt_temp.all_dates
        labels = self._make_regime_labels(dates, "mean_reversion")

        strat = AdaptiveStrategy(labels)
        bt = PortfolioBacktester(hist_dict, strat, initial_cash=10_000)
        res_adaptive = bt.run()

        # Compare with pure MR
        bt_mr = PortfolioBacktester(hist_dict, IndependentMeanReversion(20, 2.0),
                                     initial_cash=10_000)
        res_mr = bt_mr.run()

        assert res_adaptive.final_value == pytest.approx(res_mr.final_value)

    def test_relative_strength_falls_through_to_ew(self, hist_dict):
        """relative_strength regime should be treated as equal_weight."""
        bt_temp = PortfolioBacktester(hist_dict, EqualWeightRebalance(),
                                      initial_cash=10_000)
        dates = bt_temp.all_dates

        labels_rs = self._make_regime_labels(dates, "relative_strength")
        labels_ew = self._make_regime_labels(dates, "equal_weight")

        strat_rs = AdaptiveStrategy(labels_rs)
        strat_ew = AdaptiveStrategy(labels_ew)

        res_rs = PortfolioBacktester(hist_dict, strat_rs, initial_cash=10_000).run()
        res_ew = PortfolioBacktester(hist_dict, strat_ew, initial_cash=10_000).run()

        assert res_rs.final_value == pytest.approx(res_ew.final_value)

    def test_all_equal_weight(self, hist_dict):
        """If every day is equal_weight, should match EW strategy."""
        bt_temp = PortfolioBacktester(hist_dict, EqualWeightRebalance(),
                                      initial_cash=10_000)
        dates = bt_temp.all_dates
        labels = self._make_regime_labels(dates, "equal_weight")

        strat = AdaptiveStrategy(labels)
        bt = PortfolioBacktester(hist_dict, strat, initial_cash=10_000)
        res_adaptive = bt.run()

        bt_ew = PortfolioBacktester(hist_dict, EqualWeightRebalance(0.05),
                                     initial_cash=10_000)
        res_ew = bt_ew.run()

        assert res_adaptive.final_value == pytest.approx(res_ew.final_value)

    def test_regime_switch_mid_backtest(self, hist_dict):
        """Switching regime mid-backtest should produce valid results."""
        bt_temp = PortfolioBacktester(hist_dict, EqualWeightRebalance(),
                                      initial_cash=10_000)
        dates = bt_temp.all_dates
        mid = len(dates) // 2

        labels = pd.Series("equal_weight", index=dates)
        for d in dates[mid:]:
            labels[d] = "mean_reversion"

        strat = AdaptiveStrategy(labels)
        bt = PortfolioBacktester(hist_dict, strat, initial_cash=10_000)
        res = bt.run()

        assert res.final_value > 0
        assert len(res.daily_values) == len(dates)

    def test_name(self):
        strat = AdaptiveStrategy(regime_labels=pd.Series(dtype=str))
        assert strat.name == "Adaptive (EW + MR)"

    def test_sub_strategies_initialized(self):
        strat = AdaptiveStrategy(regime_labels=pd.Series(dtype=str))
        assert isinstance(strat.ew, EqualWeightRebalance)
        assert isinstance(strat.mr, IndependentMeanReversion)


# =====================================================================
# Conservation of Money --- Adaptive Strategy
# =====================================================================

class TestAdaptiveConservation:

    def test_conservation_all_regimes(self):
        """Accounting identity: cash + equity == total_value every day."""
        hist_dict = make_multi_stock_hist(n_tickers=4, n_days=60)
        bt_temp = PortfolioBacktester(hist_dict, EqualWeightRebalance(),
                                      initial_cash=10_000)
        dates = bt_temp.all_dates

        # Mix of regimes
        labels = pd.Series("equal_weight", index=dates)
        for i, d in enumerate(dates):
            if i % 5 == 0:
                labels[d] = "mean_reversion"
            elif i % 7 == 0:
                labels[d] = "relative_strength"

        strat = AdaptiveStrategy(labels)
        bt = PortfolioBacktester(hist_dict, strat, initial_cash=10_000,
                                  cash_reserve_pct=0.05)
        res = bt.run()

        for i in range(len(res.daily_values)):
            cash = res.daily_cash.iloc[i]
            equity = sum(
                res.daily_positions[t].iloc[i] * res.daily_prices[t].iloc[i]
                for t in res.tickers
            )
            total = res.daily_values.iloc[i]
            assert cash + equity == pytest.approx(total, abs=0.01), \
                f"Day {i}: cash={cash:.2f} + equity={equity:.2f} != total={total:.2f}"

    def test_no_negative_cash(self):
        """Cash should never go negative."""
        hist_dict = make_multi_stock_hist(n_tickers=3, n_days=40)
        bt_temp = PortfolioBacktester(hist_dict, EqualWeightRebalance(),
                                      initial_cash=10_000)
        dates = bt_temp.all_dates
        labels = self._alternating_regimes(dates)

        strat = AdaptiveStrategy(labels)
        bt = PortfolioBacktester(hist_dict, strat, initial_cash=10_000)
        res = bt.run()

        assert all(res.daily_cash >= -0.001)

    def test_no_negative_shares(self):
        """Shares should never go negative."""
        hist_dict = make_multi_stock_hist(n_tickers=3, n_days=40)
        bt_temp = PortfolioBacktester(hist_dict, EqualWeightRebalance(),
                                      initial_cash=10_000)
        dates = bt_temp.all_dates
        labels = self._alternating_regimes(dates)

        strat = AdaptiveStrategy(labels)
        bt = PortfolioBacktester(hist_dict, strat, initial_cash=10_000)
        res = bt.run()

        for t in res.tickers:
            assert all(res.daily_positions[t] >= -0.001)

    @staticmethod
    def _alternating_regimes(dates):
        regimes = ["equal_weight", "mean_reversion", "relative_strength"]
        return pd.Series(
            [regimes[i % len(regimes)] for i in range(len(dates))],
            index=dates,
        )


# =====================================================================
# End-to-End Backtest
# =====================================================================

class TestAdaptiveEndToEnd:

    def test_five_tickers_mixed_regimes(self):
        """5-ticker backtest with mixed regime labels runs to completion."""
        hist_dict = make_multi_stock_hist(n_tickers=5, n_days=80)
        bt_temp = PortfolioBacktester(hist_dict, EqualWeightRebalance(),
                                      initial_cash=50_000)
        dates = bt_temp.all_dates

        labels = pd.Series("equal_weight", index=dates)
        # First 20 days: MR, middle: EW, last 20: RS
        for d in dates[:20]:
            labels[d] = "mean_reversion"
        for d in dates[-20:]:
            labels[d] = "relative_strength"

        strat = AdaptiveStrategy(labels)
        bt = PortfolioBacktester(hist_dict, strat, initial_cash=50_000,
                                  cash_reserve_pct=0.05)
        res = bt.run()

        assert len(res.daily_values) == len(dates)
        assert res.final_value > 0
        assert len(res.tickers) == 5

    def test_adaptive_result_has_all_fields(self):
        """Result should have all expected attributes."""
        hist_dict = make_multi_stock_hist(n_tickers=2, n_days=30)
        strat = AdaptiveStrategy(regime_labels=pd.Series(dtype=str))
        bt = PortfolioBacktester(hist_dict, strat, initial_cash=10_000)
        res = bt.run()

        assert hasattr(res, "strategy_name")
        assert hasattr(res, "daily_values")
        assert hasattr(res, "daily_cash")
        assert hasattr(res, "daily_positions")
        assert hasattr(res, "daily_prices")
        assert hasattr(res, "trades")
        assert hasattr(res, "total_return")
        assert hasattr(res, "sharpe_ratio")
        assert hasattr(res, "max_drawdown")

    @pytest.mark.parametrize("regime", [
        "equal_weight", "mean_reversion", "relative_strength",
    ])
    def test_single_regime_completes(self, regime):
        """Each single-regime configuration runs without error."""
        hist_dict = make_multi_stock_hist(n_tickers=3, n_days=40)
        bt_temp = PortfolioBacktester(hist_dict, EqualWeightRebalance(),
                                      initial_cash=10_000)
        dates = bt_temp.all_dates
        labels = pd.Series(regime, index=dates)

        strat = AdaptiveStrategy(labels)
        bt = PortfolioBacktester(hist_dict, strat, initial_cash=10_000)
        res = bt.run()

        assert res.final_value > 0
        assert len(res.daily_values) > 0


# =====================================================================
# File Structure
# =====================================================================

class TestRegimeFileStructure:

    def test_regime_adaptive_script_exists(self):
        path = os.path.join(_REPO_ROOT, "apps", "backtester", "regime.py")
        assert os.path.isfile(path)

    def test_figure_subdirectories_exist(self):
        fig_dir = os.path.join(_REPO_ROOT, "apps", "backtester", "figures")
        for sub in ["comparisons", "regime", "diagnostics"]:
            assert os.path.isdir(os.path.join(fig_dir, sub)), \
                f"Missing figure subdirectory: {sub}"

    @pytest.mark.skipif(
        not os.path.isfile(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "apps", "backtester", "figures", "comparisons",
            "adaptive_comparison.png")),
        reason="Figures not yet generated (run backtester scripts first)")
    def test_expected_figures_exist(self):
        fig_dir = os.path.join(_REPO_ROOT, "apps", "backtester", "figures")
        expected = {
            "comparisons": [
                "adaptive_comparison.png",
                "vs_spy.png",
            ],
            "regime": [
                "regime_timeline.png",
                "regime_performance.png",
                "regime_indicators.png",
            ],
            "diagnostics": [
                "inverse_vol_comparison.png",
            ],
        }
        for sub, fnames in expected.items():
            for fname in fnames:
                path = os.path.join(fig_dir, sub, fname)
                assert os.path.isfile(path), \
                    f"Missing figure: {sub}/{fname}"


# =====================================================================
# Cross-Script Consistency
# =====================================================================

class TestRegimeCrossScriptConsistency:

    def test_vix_thresholds(self):
        """VIX_HIGH and VIX_LOW should be consistent with classify_regime."""
        assert VIX_HIGH == 25
        assert VIX_LOW == 20
        assert VIX_HIGH > VIX_LOW

    def test_ticker_list_not_empty(self):
        assert len(TICKERS) == 25

    def test_ticker_list_unique(self):
        assert len(TICKERS) == len(set(TICKERS))

    def test_regime_colors_cover_all_regimes(self):
        expected = {"equal_weight", "mean_reversion", "relative_strength"}
        assert set(REGIME_COLORS.keys()) == expected

    def test_regime_labels_cover_all_regimes(self):
        expected = {"equal_weight", "mean_reversion", "relative_strength"}
        assert set(REGIME_LABELS.keys()) == expected

    def test_adaptive_uses_same_sub_strategies(self):
        """AdaptiveStrategy should use the same strategy classes as standalone."""
        strat = AdaptiveStrategy(regime_labels=pd.Series(dtype=str))
        assert type(strat.ew) is EqualWeightRebalance
        assert type(strat.mr) is IndependentMeanReversion

    def test_adaptive_is_portfolio_strategy(self):
        strat = AdaptiveStrategy(regime_labels=pd.Series(dtype=str))
        assert isinstance(strat, PortfolioStrategy)
