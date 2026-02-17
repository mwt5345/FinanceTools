"""
Monte Carlo stress test: run portfolio strategies across random 3-year windows.

Tests whether InvVol's outperformance of EW is robust across different
market regimes, or just a lucky window.

Usage:
    python stress_test.py [--iterations 10] [--window-years 3] [--cash 5000] [--seed 603]
"""

import argparse
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

from finance_tools.utils.plotting import (
    setup_style, savefig, PALETTE, FIGSIZE, plot_kde_1d,
)
from finance_tools.backtest.monte_carlo import (
    fetch_full_history, generate_windows, run_monte_carlo,
    mc_results_to_dataframe, summarize_mc_results, compute_win_rates,
)
from finance_tools.strategies.portfolio import (
    EqualWeightRebalance, InverseVolatilityWeight, InverseVolatilityGK,
    IndependentMeanReversion, RelativeStrength,
)
from finance_tools.data.market import fetch_risk_free_rate, fetch_risk_free_history

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STRESS_TEST_12 = sorted([
    "F", "INTC", "PFE", "KO", "CSCO", "DUK",
    "EMR", "AMD", "PYPL", "ON", "HAL", "FCX",
])

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures", "stress_test")

STRATEGY_COLORS = {
    "Equal Weight": PALETTE["blue"],
    "Inverse Volatility": PALETTE["red"],
    "Inv Vol (Garman-Klass)": PALETTE["purple"],
    "Independent Mean Rev. (BB 20/2)": PALETTE["green"],
    "Relative Strength (20d)": PALETTE["yellow"],
}


def _color(name: str) -> str:
    return STRATEGY_COLORS.get(name, PALETTE["grey"])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sharpe_boxplots(df: pd.DataFrame):
    """Box plots of Sharpe ratio by strategy."""
    import matplotlib.pyplot as plt

    setup_style()
    fig, ax = plt.subplots(figsize=FIGSIZE["double"])

    strategies = sorted(df["strategy"].unique())
    data = [df[df["strategy"] == s]["sharpe"].values for s in strategies]
    colors = [_color(s) for s in strategies]

    bp = ax.boxplot(data, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_xticklabels([s.replace(" ", "\n") for s in strategies], fontsize=7)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe Ratio Distribution Across Random Windows")
    ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.5)

    fig.tight_layout()
    savefig(fig, os.path.join(FIG_DIR, "sharpe_boxplots.png"))


def plot_return_distribution(df: pd.DataFrame):
    """Overlaid KDE of annualized returns by strategy."""
    import matplotlib.pyplot as plt

    setup_style()
    fig, ax = plt.subplots(figsize=FIGSIZE["double"])

    strategies = sorted(df["strategy"].unique())
    # Shared x-range across all strategies
    all_returns = df["ann_return"].values
    lo = all_returns.min() - 0.05
    hi = all_returns.max() + 0.05
    x_range = np.linspace(lo, hi, 300)

    for s in strategies:
        vals = df[df["strategy"] == s]["ann_return"].values
        if len(vals) < 2:
            continue
        plot_kde_1d(vals, ax, color=_color(s), label=s,
                    peak_normalize=True, x_range=x_range, fill=True,
                    lw=1.5)

    ax.set_xlabel("Annualized Return")
    ax.set_ylabel("Density (peak-normalized)")
    ax.set_title("Return Distribution Across Random Windows")
    ax.legend(fontsize=6)

    fig.tight_layout()
    savefig(fig, os.path.join(FIG_DIR, "return_distribution.png"))


def plot_risk_return_scatter(df: pd.DataFrame):
    """(Volatility, Return) scatter colored by strategy."""
    import matplotlib.pyplot as plt

    setup_style()
    fig, ax = plt.subplots(figsize=FIGSIZE["single_square"])

    strategies = sorted(df["strategy"].unique())
    markers = ["o", "s", "^", "D"]

    for i, s in enumerate(strategies):
        sub = df[df["strategy"] == s]
        ax.scatter(sub["ann_vol"], sub["ann_return"],
                   color=_color(s), marker=markers[i % len(markers)],
                   s=30, alpha=0.8, label=s, edgecolors="white", linewidths=0.3)

    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.set_title("Risk--Return Scatter")
    ax.legend(fontsize=5, loc="best")

    fig.tight_layout()
    savefig(fig, os.path.join(FIG_DIR, "risk_return_scatter.png"))


def plot_drawdown_boxplots(df: pd.DataFrame):
    """Box plots of max drawdown by strategy."""
    import matplotlib.pyplot as plt

    setup_style()
    fig, ax = plt.subplots(figsize=FIGSIZE["double"])

    strategies = sorted(df["strategy"].unique())
    data = [df[df["strategy"] == s]["max_dd"].values for s in strategies]
    colors = [_color(s) for s in strategies]

    bp = ax.boxplot(data, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_xticklabels([s.replace(" ", "\n") for s in strategies], fontsize=7)
    ax.set_ylabel("Max Drawdown")
    ax.set_title("Max Drawdown Distribution Across Random Windows")

    fig.tight_layout()
    savefig(fig, os.path.join(FIG_DIR, "drawdown_boxplots.png"))


def plot_summary_table(summary: pd.DataFrame, win_matrix: pd.DataFrame):
    """Render summary stats + pairwise win-rate matrix as a figure."""
    import matplotlib.pyplot as plt

    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5),
                              gridspec_kw={"width_ratios": [3, 2]})

    # --- Left panel: summary stats ---
    ax_table = axes[0]
    ax_table.axis("off")

    # Build table data
    cols = ["Sharpe\n(mean)", "Sharpe\n(med)", "Return\n(mean)",
            "Vol\n(mean)", "MaxDD\n(mean)", "Win\nRate"]
    rows = []
    row_labels = []
    for strat in summary.index:
        row_labels.append(strat)
        sharpe_mean = f"{summary.loc[strat, 'sharpe_mean']:.2f}"
        sharpe_med = f"{summary.loc[strat, 'sharpe_median']:.2f}"
        ret_mean = f"{summary.loc[strat, 'ann_return_mean']:.1%}"
        vol_mean = f"{summary.loc[strat, 'ann_vol_mean']:.1%}"
        dd_mean = f"{summary.loc[strat, 'max_dd_mean']:.1%}"
        wr = f"{summary.loc[strat, 'win_rate']:.0%}"
        rows.append([sharpe_mean, sharpe_med, ret_mean, vol_mean, dd_mean, wr])

    tbl = ax_table.table(
        cellText=rows, colLabels=cols, rowLabels=row_labels,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 1.4)
    ax_table.set_title("Summary Statistics", fontsize=9, pad=12)

    # --- Right panel: pairwise win-rate matrix ---
    ax_win = axes[1]
    ax_win.axis("off")

    strats = list(win_matrix.index)
    short_names = [s[:12] for s in strats]
    win_data = []
    for s_a in strats:
        row = []
        for s_b in strats:
            val = win_matrix.loc[s_a, s_b]
            row.append("---" if pd.isna(val) else f"{val:.0%}")
        win_data.append(row)

    tbl2 = ax_win.table(
        cellText=win_data, colLabels=short_names, rowLabels=short_names,
        loc="center", cellLoc="center",
    )
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(7)
    tbl2.scale(1.0, 1.4)
    ax_win.set_title("Pairwise Sharpe Win Rate", fontsize=9, pad=12)

    fig.tight_layout()
    savefig(fig, os.path.join(FIG_DIR, "summary_table.png"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo stress test across random time windows")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of random windows (default: 10)")
    parser.add_argument("--window-years", type=float, default=3.0,
                        help="Window duration in years (default: 3)")
    parser.add_argument("--cash", type=float, default=5_000.0,
                        help="Initial cash per window (default: $5,000)")
    parser.add_argument("--seed", type=int, default=603,
                        help="Random seed (default: 603)")
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)

    # --- Step 1: Compute 15-year date range ---
    end_date = date.today()
    start_date = end_date - timedelta(days=int(15 * 365.25))
    print(f"Fetching 15 years of data: {start_date} to {end_date}")
    print(f"Tickers: {', '.join(STRESS_TEST_12)}")

    # --- Step 2: Fetch all data once ---
    hist_dict = fetch_full_history(STRESS_TEST_12, start_date, end_date)
    available = sorted(hist_dict.keys())
    print(f"Got data for {len(available)} / {len(STRESS_TEST_12)} tickers: "
          f"{', '.join(available)}")
    for t in available:
        print(f"  {t}: {len(hist_dict[t])} trading days")

    if len(available) < 2:
        print("Need at least 2 tickers with data. Exiting.")
        sys.exit(1)

    # --- Step 3: Generate random windows ---
    windows = generate_windows(
        hist_dict,
        window_years=args.window_years,
        n_iterations=args.iterations,
        seed=args.seed,
    )
    print(f"\nGenerated {len(windows)} windows "
          f"({args.window_years}y each, seed={args.seed}):")
    for w in windows:
        print(f"  #{w.iteration}: {w.start} to {w.end}")

    # --- Step 4: Create strategies ---
    strategies = [
        EqualWeightRebalance(threshold=0.05),
        InverseVolatilityWeight(threshold=0.05),
        InverseVolatilityGK(threshold=0.03, cash_reserve_pct=0.02),
        IndependentMeanReversion(20, 2.0),
        RelativeStrength(20),
    ]

    # --- Step 5: Fetch risk-free rate ---
    rf_rate = fetch_risk_free_rate()
    rf_daily = fetch_risk_free_history(start_date, end_date)
    if len(rf_daily) > 0:
        avg_rf = float(rf_daily.mean())
        print(f"\nRisk-free rate: avg {avg_rf:.2%} over period (spot {rf_rate:.2%})")
    else:
        rf_daily = None
        print(f"\nRisk-free rate (3-mo T-bill): {rf_rate:.2%}")

    # --- Step 6: Run MC ---
    print(f"\nRunning {len(windows)} windows x {len(strategies)} strategies "
          f"= {len(windows) * len(strategies)} backtests...")
    mc_results = run_monte_carlo(
        hist_dict, strategies, windows,
        initial_cash=args.cash,
        cash_reserve_pct=0.05,
        rf_rate=rf_rate,
        rf_history=rf_daily,
    )
    print(f"Collected {len(mc_results)} results.")

    # --- Step 7: Aggregate ---
    df = mc_results_to_dataframe(mc_results)
    summary = summarize_mc_results(df)
    win_matrix = compute_win_rates(df)

    # --- Step 8: Print summary ---
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    for strat in summary.index:
        s = summary.loc[strat]
        print(f"\n  {strat}:")
        print(f"    Sharpe:   {s['sharpe_mean']:.2f} mean, "
              f"{s['sharpe_median']:.2f} median, "
              f"{s['sharpe_std']:.2f} std")
        print(f"    Return:   {s['ann_return_mean']:.1%} mean, "
              f"{s['ann_return_std']:.1%} std")
        print(f"    Vol:      {s['ann_vol_mean']:.1%} mean")
        print(f"    Max DD:   {s['max_dd_mean']:.1%} mean")
        print(f"    Win rate: {s['win_rate']:.0%}")

    print(f"\n{'='*70}")
    print("PAIRWISE SHARPE WIN RATES (row beats column)")
    print(f"{'='*70}")
    # Formatted print
    strats = list(win_matrix.index)
    header = f"{'':>25s}" + "".join(f"{s[:12]:>14s}" for s in strats)
    print(header)
    for s_a in strats:
        row_str = f"{s_a:>25s}"
        for s_b in strats:
            val = win_matrix.loc[s_a, s_b]
            cell = "---" if pd.isna(val) else f"{val:.0%}"
            row_str += f"{cell:>14s}"
        print(row_str)

    # --- Step 9: Generate figures ---
    print("\nGenerating figures...")
    plot_sharpe_boxplots(df)
    plot_return_distribution(df)
    plot_risk_return_scatter(df)
    plot_drawdown_boxplots(df)
    plot_summary_table(summary, win_matrix)

    print("Done!")


if __name__ == "__main__":
    main()
