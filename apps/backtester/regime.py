"""
Regime-adaptive portfolio strategy analysis.

Computes market regime indicators and switches between strategies:
  - Default: Equal Weight (steady diversified rebalancing)
  - VIX > 25: Mean Reversion (buy the dip in stressed markets)
  - VIX < 20 AND SPY > 200MA: Relative Strength (ride momentum in calm bulls)

Usage:
    python regime.py
"""

import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from finance_tools.utils.plotting import setup_style, savefig, PALETTE, FIGSIZE
from finance_tools.backtest.portfolio import PortfolioBacktester, PortfolioStrategy
from finance_tools.strategies.portfolio import (
    EqualWeightRebalance, IndependentMeanReversion, RelativeStrength,
)
from finance_tools.data.market import fetch_risk_free_rate, fetch_risk_free_history
from finance_tools.data.universe import REGIME_25
from finance_tools.data.results import ResultsStore

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
FIG_COMP = os.path.join(FIG_DIR, "comparisons")
FIG_REGIME = os.path.join(FIG_DIR, "regime")
FIG_DIAG = os.path.join(FIG_DIR, "diagnostics")

TICKERS = REGIME_25

# Regime thresholds
VIX_HIGH = 25   # Above -> stressed -> Mean Reversion
VIX_LOW = 20    # Below + bull trend -> calm -> Relative Strength


# =====================================================================
# Data & Indicators
# =====================================================================

def fetch_data(period="10y"):
    """Download stock data + SPY + VIX."""
    print(f"Downloading {len(TICKERS)} stocks + SPY + VIX ({period})...")
    stock_data = {}
    for t in TICKERS:
        h = yf.Ticker(t).history(period=period)
        if not h.empty:
            stock_data[t] = h
            print(f"  {t}: {len(h)} days")
        else:
            print(f"  WARNING: {t} returned no data, skipping")

    spy = yf.Ticker("SPY").history(period=period)
    vix = yf.Ticker("^VIX").history(period=period)
    print(f"  SPY: {len(spy)} days")
    print(f"  VIX: {len(vix)} days")
    return stock_data, spy, vix


def _normalize_index(df):
    """Strip timezone and normalize to midnight."""
    df = df.copy()
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep="first")]
    return df


def compute_indicators(stock_data, spy, vix):
    """Compute daily regime indicators on a normalized date index."""
    spy = _normalize_index(spy)
    vix = _normalize_index(vix)

    # VIX (5-day smoothed)
    vix_smooth = vix["Close"].rolling(5, min_periods=1).mean()

    # SPY trend: close vs 200-day MA
    spy_close = spy["Close"]
    spy_ma200 = spy_close.rolling(200, min_periods=200).mean()
    spy_bull = (spy_close > spy_ma200).astype(float)

    # Cross-sectional dispersion (20-day rolling mean of daily cross-sectional std)
    stock_norm = {t: _normalize_index(df) for t, df in stock_data.items()}
    all_dates = sorted(set().union(*(set(df.index) for df in stock_norm.values())))
    closes = pd.DataFrame({
        t: df["Close"].reindex(all_dates) for t, df in stock_norm.items()
    })
    returns = closes.pct_change()
    dispersion = returns.std(axis=1).rolling(20, min_periods=5).mean()

    # Average pairwise correlation (60-day rolling)
    n = len(all_dates)
    avg_corr = pd.Series(np.nan, index=closes.index)
    for i in range(60, n):
        window = returns.iloc[i - 60:i].dropna(axis=1, how="any")
        if len(window.columns) < 2:
            continue
        cm = window.corr().values
        mask = np.triu(np.ones_like(cm, dtype=bool), k=1)
        avg_corr.iloc[i] = np.nanmean(cm[mask])

    return pd.DataFrame({
        "vix": vix_smooth,
        "spy_close": spy_close,
        "spy_ma200": spy_ma200,
        "spy_bull": spy_bull,
        "dispersion": dispersion,
        "avg_corr": avg_corr,
    })


def classify_regime(indicators):
    """Classify each day: equal_weight / mean_reversion / relative_strength."""
    regime = pd.Series("equal_weight", index=indicators.index)

    # Mean Reversion: high VIX (stressed market, correlated selloff)
    stressed = indicators["vix"] > VIX_HIGH
    regime[stressed] = "mean_reversion"

    # Relative Strength overrides: calm bull market
    # (applied second so calm-bull wins if somehow both trigger -- shouldn't happen)
    calm_bull = (indicators["vix"] < VIX_LOW) & (indicators["spy_bull"] == 1.0)
    regime[calm_bull] = "relative_strength"

    return regime


# =====================================================================
# Adaptive Strategy
# =====================================================================

class AdaptiveStrategy(PortfolioStrategy):
    """
    Defaults to Equal Weight. Switches to Mean Reversion in stressed markets:
      - stressed (VIX > 25) -> Mean Reversion
      - otherwise -> Equal Weight
    """
    name = "Adaptive (EW + MR)"

    def __init__(self, regime_labels):
        self.regime_labels = regime_labels
        self.ew = EqualWeightRebalance(threshold=0.05)
        self.mr = IndependentMeanReversion(20, 2.0)

    def decide(self, day, history, portfolio):
        date = next(iter(day.values())).name
        if hasattr(date, "normalize"):
            date = date.normalize()

        regime = self.regime_labels.get(date, "equal_weight")

        if regime == "mean_reversion":
            return self.mr.decide(day, history, portfolio)
        else:
            return self.ew.decide(day, history, portfolio)


# =====================================================================
# Plotting
# =====================================================================

REGIME_COLORS = {
    "equal_weight": PALETTE["blue"],
    "mean_reversion": PALETTE["red"],
    "relative_strength": PALETTE["green"],
}
REGIME_LABELS = {
    "equal_weight": "Equal Weight",
    "mean_reversion": "Mean Reversion",
    "relative_strength": "Rel. Strength",
}
STRATEGY_COLORS = [
    PALETTE["blue"], PALETTE["red"], PALETTE["green"], PALETTE["purple"],
]


def _add_regime_bg(ax, regime_labels):
    """Add regime background shading (grouped into contiguous spans)."""
    prev = None
    start = None
    for date, regime in regime_labels.items():
        if regime != prev:
            if prev is not None:
                ax.axvspan(start, date, alpha=0.12,
                           color=REGIME_COLORS.get(prev, "grey"), linewidth=0)
            start = date
            prev = regime
    if prev is not None:
        ax.axvspan(start, regime_labels.index[-1], alpha=0.12,
                   color=REGIME_COLORS.get(prev, "grey"), linewidth=0)


def plot_comparison(results):
    """4-strategy portfolio value comparison."""
    import matplotlib.pyplot as plt

    setup_style()
    fig, ax = plt.subplots(figsize=FIGSIZE["double"])

    for res, color in zip(results, STRATEGY_COLORS):
        ax.plot(res.daily_values.index, res.daily_values.values,
                color=color, lw=1.2, label=res.strategy_name)

    ax.set_ylabel("Portfolio Value (\\$)")
    ax.set_title("25-Stock Diversified --- Strategy Comparison")
    ax.legend(fontsize=7)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    savefig(fig, os.path.join(FIG_COMP, "adaptive_comparison.png"))


def plot_regime_timeline(indicators, regime_labels, results):
    """3-panel: adaptive portfolio + VIX + SPY, with regime backgrounds."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    setup_style()
    fig, axes = plt.subplots(3, 1, figsize=(FIGSIZE["double"][0], 7),
                             sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1]})

    for ax in axes:
        _add_regime_bg(ax, regime_labels)

    # Panel 1: Adaptive portfolio value
    ax = axes[0]
    res_adaptive = results[-1]
    ax.plot(res_adaptive.daily_values.index,
            res_adaptive.daily_values.values,
            color=PALETTE["purple"], lw=1.0)
    ax.set_ylabel("Portfolio Value (\\$)")
    ax.set_title("Adaptive Strategy with Regime Background")
    legend_patches = [
        Patch(facecolor=REGIME_COLORS[r], alpha=0.3, label=REGIME_LABELS[r])
        for r in ["equal_weight", "mean_reversion", "relative_strength"]
    ]
    ax.legend(handles=legend_patches, fontsize=6, loc="upper left")

    # Panel 2: VIX
    ax = axes[1]
    vix = indicators["vix"].dropna()
    ax.plot(vix.index, vix.values, color="black", lw=0.6)
    ax.axhline(VIX_HIGH, color=PALETTE["red"], ls="--", lw=0.7,
               label=f"VIX = {VIX_HIGH}")
    ax.axhline(VIX_LOW, color=PALETTE["green"], ls="--", lw=0.7,
               label=f"VIX = {VIX_LOW}")
    ax.set_ylabel("VIX (5d MA)")
    ax.legend(fontsize=6)

    # Panel 3: SPY vs 200-day MA
    ax = axes[2]
    spy_c = indicators["spy_close"].dropna()
    spy_m = indicators["spy_ma200"].dropna()
    ax.plot(spy_c.index, spy_c.values, color="black", lw=0.6, label="SPY")
    ax.plot(spy_m.index, spy_m.values, color=PALETTE["grey"], lw=0.6,
            ls="--", label="200d MA")
    ax.set_ylabel("SPY Price (\\$)")
    ax.legend(fontsize=6)
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    savefig(fig, os.path.join(FIG_REGIME, "regime_timeline.png"))


def plot_regime_performance(results, regime_labels):
    """Grouped bar chart: per-regime annualized Sharpe for each strategy."""
    import matplotlib.pyplot as plt

    setup_style()
    regimes = ["equal_weight", "mean_reversion", "relative_strength"]
    fig, ax = plt.subplots(figsize=FIGSIZE["double"])

    x = np.arange(len(regimes))
    width = 0.18

    for j, res in enumerate(results):
        sharpes = []
        for regime in regimes:
            regime_dates = regime_labels[regime_labels == regime].index
            common = res.daily_values.index.intersection(regime_dates)
            if len(common) < 5:
                sharpes.append(0.0)
                continue
            rets = res.daily_values.loc[common].pct_change().dropna()
            if len(rets) < 2 or rets.std() == 0:
                sharpes.append(0.0)
            else:
                sharpes.append(rets.mean() / rets.std() * np.sqrt(252))
        ax.bar(x + j * width, sharpes, width, color=STRATEGY_COLORS[j],
               label=res.strategy_name, alpha=0.85)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([REGIME_LABELS[r] for r in regimes])
    ax.set_ylabel("Annualized Sharpe Ratio")
    ax.set_title("Strategy Performance by Market Regime")
    ax.legend(fontsize=6)
    ax.axhline(0, color="black", lw=0.5)
    fig.tight_layout()
    savefig(fig, os.path.join(FIG_REGIME, "regime_performance.png"))


def plot_indicators(indicators, regime_labels):
    """4-panel: VIX, SPY trend, dispersion, avg correlation."""
    import matplotlib.pyplot as plt

    setup_style()
    fig, axes = plt.subplots(4, 1, figsize=(FIGSIZE["double"][0], 9),
                             sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1]})

    for ax in axes:
        _add_regime_bg(ax, regime_labels)

    # VIX
    ax = axes[0]
    vix = indicators["vix"].dropna()
    ax.plot(vix.index, vix.values, color="black", lw=0.6)
    ax.axhline(VIX_HIGH, color=PALETTE["red"], ls="--", lw=0.7)
    ax.axhline(VIX_LOW, color=PALETTE["green"], ls="--", lw=0.7)
    ax.set_ylabel("VIX (5d MA)")
    ax.set_title("Regime Indicators")

    # SPY trend
    ax = axes[1]
    spy_c = indicators["spy_close"].dropna()
    spy_m = indicators["spy_ma200"].dropna()
    ax.plot(spy_c.index, spy_c.values, color="black", lw=0.6, label="SPY")
    ax.plot(spy_m.index, spy_m.values, color=PALETTE["grey"], lw=0.6,
            ls="--", label="200d MA")
    ax.set_ylabel("SPY (\\$)")
    ax.legend(fontsize=6)

    # Dispersion
    ax = axes[2]
    disp = indicators["dispersion"].dropna()
    ax.plot(disp.index, disp.values, color=PALETTE["cyan"], lw=0.6)
    ax.set_ylabel("Dispersion")

    # Average correlation
    ax = axes[3]
    corr = indicators["avg_corr"].dropna()
    ax.plot(corr.index, corr.values, color=PALETTE["yellow"], lw=0.6)
    ax.axhline(0.5, color="black", ls=":", lw=0.5)
    ax.set_ylabel("Avg Pairwise Corr")
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    savefig(fig, os.path.join(FIG_REGIME, "regime_indicators.png"))


def plot_vs_spy(results, spy):
    """Normalized growth comparison: all strategies vs S&P 500 (SPY)."""
    import matplotlib.pyplot as plt

    setup_style()
    fig, ax = plt.subplots(figsize=FIGSIZE["double"])

    spy = _normalize_index(spy)

    # Normalize SPY growth, aligned to strategy dates
    dates = results[0].daily_values.index
    spy_close = spy["Close"].reindex(dates).ffill().bfill()
    spy_growth = spy_close / spy_close.iloc[0]

    # Plot each strategy's growth relative to SPY
    for res, color in zip(results, STRATEGY_COLORS):
        strat_growth = res.daily_values / res.daily_values.iloc[0]
        relative = strat_growth / spy_growth
        ax.plot(dates, relative.values, color=color, lw=1.0,
                label=res.strategy_name)

    ax.set_ylabel("Growth Relative to S\\&P 500")
    ax.set_title("25-Stock Diversified vs.\\ S\\&P 500")
    ax.legend(fontsize=7)
    ax.tick_params(axis="x", rotation=30)
    ax.axhline(1.0, color="black", lw=0.6, ls="--", label="_nolegend_")
    fig.tight_layout()
    savefig(fig, os.path.join(FIG_COMP, "vs_spy.png"))


def plot_inverse_vol_comparison(stock_data):
    """Compare EW with inverse-vol vs. most-underweight buy priority."""
    import matplotlib.pyplot as plt

    setup_style()

    # Run EW with inverse-vol (current default)
    strat_iv = EqualWeightRebalance(threshold=0.05, vol_lookback=60)
    bt_iv = PortfolioBacktester(stock_data, strat_iv,
                                 initial_cash=100_000, cash_reserve_pct=0.05)
    res_iv = bt_iv.run()

    class EWMostUnderweight(EqualWeightRebalance):
        name = "EW (Most Underweight First)"

    strat_uw = EWMostUnderweight(threshold=0.05)
    bt_uw = PortfolioBacktester(stock_data, strat_uw,
                                 initial_cash=100_000, cash_reserve_pct=0.05)
    res_uw = bt_uw.run()

    # 2-panel figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIGSIZE["double"][0], 6),
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Panel 1: Portfolio value
    ax1.plot(res_iv.daily_values.index, res_iv.daily_values.values,
             color=PALETTE["blue"], lw=1.0, label="Inverse Volatility")
    ax1.plot(res_uw.daily_values.index, res_uw.daily_values.values,
             color=PALETTE["red"], lw=1.0, ls="--", label="Most Underweight")
    ax1.set_ylabel("Portfolio Value (\\$)")
    ax1.set_title("Equal Weight Buy Priority --- Inverse Vol vs.\\ Most Underweight")
    ax1.legend(fontsize=7)

    iv_ret = f"{res_iv.total_return:.1%}"
    uw_ret = f"{res_uw.total_return:.1%}"
    iv_sharpe = f"{res_iv.sharpe_ratio:.2f}"
    uw_sharpe = f"{res_uw.sharpe_ratio:.2f}"
    iv_dd = f"{res_iv.max_drawdown:.1%}"
    uw_dd = f"{res_uw.max_drawdown:.1%}"
    text = (f"Inverse Vol: {iv_ret} return, {iv_sharpe} Sharpe, {iv_dd} DD\n"
            f"Underweight: {uw_ret} return, {uw_sharpe} Sharpe, {uw_dd} DD")
    ax1.text(0.02, 0.95, text, transform=ax1.transAxes, fontsize=6,
             verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Panel 2: Difference (inverse vol - underweight)
    diff = res_iv.daily_values - res_uw.daily_values
    ax2.fill_between(diff.index, 0, diff.values,
                     where=diff.values >= 0, color=PALETTE["blue"], alpha=0.3)
    ax2.fill_between(diff.index, 0, diff.values,
                     where=diff.values < 0, color=PALETTE["red"], alpha=0.3)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("Difference (\\$)")
    ax2.set_xlabel("")
    ax2.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    savefig(fig, os.path.join(FIG_DIAG, "inverse_vol_comparison.png"))

    print(f"\n  Inverse Vol:    ${res_iv.final_value:,.0f}  "
          f"Sharpe={res_iv.sharpe_ratio:.2f}  DD={res_iv.max_drawdown:.1%}")
    print(f"  Underweight:    ${res_uw.final_value:,.0f}  "
          f"Sharpe={res_uw.sharpe_ratio:.2f}  DD={res_uw.max_drawdown:.1%}")


# =====================================================================
# Main
# =====================================================================

def main():
    for d in [FIG_COMP, FIG_REGIME, FIG_DIAG]:
        os.makedirs(d, exist_ok=True)

    # 1. Fetch data
    stock_data, spy, vix = fetch_data("10y")

    # 2. Compute indicators
    print("\nComputing regime indicators...")
    indicators = compute_indicators(stock_data, spy, vix)

    # 3. Classify regimes
    regime_labels = classify_regime(indicators)
    counts = regime_labels.value_counts()
    total = len(regime_labels)
    print(f"\nRegime breakdown ({total} trading days):")
    for regime in ["equal_weight", "mean_reversion", "relative_strength"]:
        n = counts.get(regime, 0)
        print(f"  {REGIME_LABELS[regime]:20s} {n:5d} days ({n / total:.1%})")

    # 4. Fetch risk-free rate
    rf_rate = fetch_risk_free_rate()
    rf_start = date.today() - timedelta(days=int(10 * 365.25))
    rf_daily = fetch_risk_free_history(rf_start)
    if len(rf_daily) > 0:
        avg_rf = float(rf_daily.mean())
        print(f"\nRisk-free rate: avg {avg_rf:.2%} over period (spot {rf_rate:.2%})")
    else:
        rf_daily = None
        print(f"\nRisk-free rate (3-mo T-bill): {rf_rate:.2%}")

    # 5. Run backtests
    strategies = [
        EqualWeightRebalance(threshold=0.05),
        IndependentMeanReversion(20, 2.0),
        RelativeStrength(20),
        AdaptiveStrategy(regime_labels),
    ]

    results = []
    print(f"\n{'=' * 60}")
    for strat in strategies:
        bt = PortfolioBacktester(
            stock_data, strat, initial_cash=100_000, cash_reserve_pct=0.05,
        )
        res = bt.run()
        res.rf_rate = rf_rate
        if rf_daily is not None:
            res.rf_daily = rf_daily
        results.append(res)
        print(res.summary())
        print(f"  Days:          {len(res.daily_values)}")
        print(f"{'=' * 60}")

    # 6. Save results
    store = ResultsStore()
    for res in results:
        run_id = store.save(res, runner_script="regime.py")
        print(f"  Saved {res.strategy_name} as run #{run_id}")
    store.close()

    # 7. Figures
    print("\nGenerating figures...")
    plot_comparison(results)
    plot_regime_timeline(indicators, regime_labels, results)
    plot_regime_performance(results, regime_labels)
    plot_indicators(indicators, regime_labels)
    plot_vs_spy(results, spy)
    plot_inverse_vol_comparison(stock_data)

    print("Done!")


if __name__ == "__main__":
    main()
