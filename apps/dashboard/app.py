"""
Backtest Dashboard

Interactive Streamlit app for comparing backtest results stored in SQLite.

Usage:
    streamlit run app.py
"""

import sys
import os
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from finance_tools.data.results import ResultsStore
from finance_tools.data.universe import get_sector, SECTORS

# ---------------------------------------------------------------------------
# Tol Bright palette (colorblind-safe, matches plotting.py)
# ---------------------------------------------------------------------------
TOL_BRIGHT = [
    "#4477AA",  # blue
    "#66CCEE",  # cyan
    "#228833",  # green
    "#CCBB44",  # yellow
    "#EE6677",  # red
    "#AA3377",  # purple
    "#BBBBBB",  # grey
]

SECTOR_COLORS = {}
for i, sector in enumerate(sorted(SECTORS.keys())):
    SECTOR_COLORS[sector] = TOL_BRIGHT[i % len(TOL_BRIGHT)]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Backtest Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# ---------------------------------------------------------------------------
# DB connection (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_store():
    return ResultsStore()


@st.cache_data(ttl=3600)
def fetch_spy(start: str, end: str) -> pd.DataFrame:
    """Fetch SPY OHLCV for benchmark comparison (cached 1h)."""
    hist = yf.Ticker("SPY").history(start=start, end=end)
    return hist


def run_label(run: dict) -> str:
    """Human-readable label for a run."""
    tickers = json.loads(run["tickers"])
    n = len(tickers)
    start = run.get("period_start", "?")
    end = run.get("period_end", "?")
    return f"#{run['run_id']}: {run['strategy_name']} ({n} tkrs, {start} to {end})"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Backtest Dashboard")

store = get_store()
all_runs = store.list_runs(limit=200)

if not all_runs:
    st.warning(
        "No backtest results in the database. "
        "Run a backtest first:\n\n"
        "```\npython apps/backtester/run.py\n```"
    )
    st.stop()

# Strategy filter
strategies = sorted({r["strategy_name"] for r in all_runs})
selected_strategies = st.sidebar.multiselect(
    "Filter by strategy", strategies, default=strategies,
)

# Filter runs by strategy
filtered_runs = [r for r in all_runs if r["strategy_name"] in selected_strategies]

if not filtered_runs:
    st.info("No runs match the selected filters.")
    st.stop()

# Run selector
run_options = {run_label(r): r["run_id"] for r in filtered_runs}
selected_labels = st.sidebar.multiselect(
    "Select runs to compare",
    list(run_options.keys()),
    default=list(run_options.keys())[:4],
)
selected_ids = [run_options[lbl] for lbl in selected_labels]

if not selected_ids:
    st.info("Select at least one run from the sidebar.")
    st.stop()

# Fetch run details
selected_runs = [store.get_run(rid) for rid in selected_ids]
selected_runs = [r for r in selected_runs if r is not None]

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Portfolio Value Over Time",
    "Metrics Comparison",
    "Per-Ticker Breakdown",
    "vs Treasuries",
    "vs SPY",
])

# ===== Tab 1: Portfolio Value Over Time =====================================
with tab1:
    st.header("Portfolio Value Over Time")

    show_cash = st.checkbox("Show cash overlay", value=False)
    show_drawdown = st.checkbox("Show drawdown", value=False)

    daily_multi = store.get_daily_values_multi(selected_ids)

    # Check which runs have daily data
    runs_with_data = [rid for rid in selected_ids if daily_multi.get(rid)]
    runs_without = [rid for rid in selected_ids if not daily_multi.get(rid)]

    if runs_without:
        labels = ", ".join(f"#{rid}" for rid in runs_without)
        st.warning(
            f"Runs {labels} have no daily time series data "
            "(saved before this feature was added). "
            "Re-run the backtest to backfill."
        )

    if runs_with_data:
        # Portfolio value chart
        fig = go.Figure()
        for i, rid in enumerate(runs_with_data):
            rows = daily_multi[rid]
            dates = [r["date"] for r in rows]
            values = [r["portfolio_value"] for r in rows]
            run = store.get_run(rid)
            color = TOL_BRIGHT[i % len(TOL_BRIGHT)]
            fig.add_trace(go.Scatter(
                x=dates, y=values, mode="lines",
                name=f"#{rid} {run['strategy_name']}",
                line=dict(color=color, width=2),
            ))
        fig.update_layout(
            yaxis_title="Portfolio Value ($)",
            xaxis_title="Date",
            hovermode="x unified",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Cash overlay
        if show_cash:
            fig_cash = go.Figure()
            for i, rid in enumerate(runs_with_data):
                rows = daily_multi[rid]
                dates = [r["date"] for r in rows]
                cash = [r["cash_value"] for r in rows]
                run = store.get_run(rid)
                color = TOL_BRIGHT[i % len(TOL_BRIGHT)]
                fig_cash.add_trace(go.Scatter(
                    x=dates, y=cash, mode="lines",
                    name=f"#{rid} {run['strategy_name']}",
                    line=dict(color=color, width=2),
                ))
            fig_cash.update_layout(
                yaxis_title="Cash ($)",
                xaxis_title="Date",
                hovermode="x unified",
                height=400,
            )
            st.plotly_chart(fig_cash, use_container_width=True)

        # Drawdown chart
        if show_drawdown:
            fig_dd = go.Figure()
            for i, rid in enumerate(runs_with_data):
                rows = daily_multi[rid]
                dates = [r["date"] for r in rows]
                values = pd.Series([r["portfolio_value"] for r in rows])
                running_max = values.cummax()
                drawdown = (values - running_max) / running_max
                run = store.get_run(rid)
                color = TOL_BRIGHT[i % len(TOL_BRIGHT)]
                fig_dd.add_trace(go.Scatter(
                    x=dates, y=drawdown.tolist(), mode="lines",
                    name=f"#{rid} {run['strategy_name']}",
                    line=dict(color=color, width=2),
                    fill="tozeroy",
                ))
            fig_dd.update_layout(
                yaxis_title="Drawdown",
                yaxis_tickformat=".0%",
                xaxis_title="Date",
                hovermode="x unified",
                height=400,
            )
            st.plotly_chart(fig_dd, use_container_width=True)
    else:
        st.info("No selected runs have daily time series data.")


# ===== Tab 2: Metrics Comparison ============================================
with tab2:
    st.header("Metrics Comparison")

    # Summary metric cards
    if selected_runs:
        sharpes = [(r["run_id"], r["sharpe_ratio"]) for r in selected_runs
                   if r.get("sharpe_ratio") is not None]
        returns = [(r["run_id"], r["annualized_return"]) for r in selected_runs
                   if r.get("annualized_return") is not None]
        drawdowns = [(r["run_id"], r["max_drawdown"]) for r in selected_runs
                     if r.get("max_drawdown") is not None]

        col1, col2, col3, col4 = st.columns(4)
        if sharpes:
            best = max(sharpes, key=lambda x: x[1])
            col1.metric("Best Sharpe", f"{best[1]:.2f}", f"Run #{best[0]}")
        if returns:
            best = max(returns, key=lambda x: x[1])
            col2.metric("Best Ann. Return", f"{best[1]:.1%}", f"Run #{best[0]}")
        if drawdowns:
            best = max(drawdowns, key=lambda x: x[1])  # least negative
            col3.metric("Shallowest DD", f"{best[1]:.1%}", f"Run #{best[0]}")
        trades = [(r["run_id"], r["n_trades"]) for r in selected_runs
                  if r.get("n_trades") is not None]
        if trades:
            most = max(trades, key=lambda x: x[1])
            col4.metric("Most Trades", f"{most[1]:,d}", f"Run #{most[0]}")

    # Comparison table
    rows = []
    for r in selected_runs:
        rows.append({
            "Run": f"#{r['run_id']}",
            "Strategy": r["strategy_name"],
            "Tickers": r["n_tickers"],
            "Period": f"{r.get('period_start', '?')} to {r.get('period_end', '?')}",
            "Days": r.get("n_trading_days", 0),
            "Initial ($)": r.get("initial_cash", 0),
            "Final ($)": r.get("final_value", 0),
            "Total Return": r.get("total_return", 0),
            "Ann. Return": r.get("annualized_return", 0),
            "Ann. Vol": r.get("annualized_volatility", 0),
            "Sharpe": r.get("sharpe_ratio", 0),
            "Max DD": r.get("max_drawdown", 0),
            "Rf Rate": r.get("rf_rate", 0),
            "Trades": r.get("n_trades", 0),
            "Dividends": r.get("n_dividends", 0),
            "Final Cash ($)": r.get("final_cash", 0),
        })

    if rows:
        df = pd.DataFrame(rows)

        # Format columns
        format_dict = {
            "Initial ($)": "${:,.0f}",
            "Final ($)": "${:,.0f}",
            "Final Cash ($)": "${:,.0f}",
            "Total Return": "{:.1%}",
            "Ann. Return": "{:.1%}",
            "Ann. Vol": "{:.1%}",
            "Sharpe": "{:.2f}",
            "Max DD": "{:.1%}",
            "Rf Rate": "{:.2%}",
            "Trades": "{:,d}",
            "Dividends": "{:,d}",
            "Days": "{:,d}",
        }

        styled = df.style.format(format_dict, na_rep="N/A")

        # Conditional formatting for key metrics
        numeric_subset = ["Sharpe", "Ann. Return", "Total Return"]
        for col in numeric_subset:
            if col in df.columns:
                styled = styled.background_gradient(
                    subset=[col], cmap="RdYlGn", axis=0,
                )
        if "Max DD" in df.columns:
            styled = styled.background_gradient(
                subset=["Max DD"], cmap="RdYlGn", axis=0,
            )

        st.dataframe(styled, use_container_width=True, hide_index=True)


# ===== Tab 3: Per-Ticker Breakdown ==========================================
with tab3:
    st.header("Per-Ticker Breakdown")

    # Limit to 4 runs for side-by-side
    compare_labels = st.multiselect(
        "Select up to 4 runs for comparison",
        selected_labels,
        default=selected_labels[:min(4, len(selected_labels))],
        max_selections=4,
    )
    compare_ids = [run_options[lbl] for lbl in compare_labels]

    if not compare_ids:
        st.info("Select at least one run above.")
    else:
        cols = st.columns(min(len(compare_ids), 4))
        for idx, rid in enumerate(compare_ids):
            with cols[idx]:
                run = store.get_run(rid)
                contribs = store.get_ticker_contributions(rid)
                st.subheader(f"#{rid}: {run['strategy_name']}")

                if not contribs:
                    st.info("No ticker data for this run.")
                    continue

                # Build DataFrame
                df_t = pd.DataFrame(contribs)
                df_t["sector"] = df_t["ticker"].apply(
                    lambda t: get_sector(t) or "Unknown"
                )
                df_t = df_t.sort_values("pnl", ascending=True)

                # Horizontal bar chart colored by sector
                fig_bar = go.Figure()
                for sector in sorted(df_t["sector"].unique()):
                    mask = df_t["sector"] == sector
                    subset = df_t[mask]
                    fig_bar.add_trace(go.Bar(
                        y=subset["ticker"],
                        x=subset["pnl"],
                        name=sector,
                        orientation="h",
                        marker_color=SECTOR_COLORS.get(sector, "#999999"),
                    ))
                fig_bar.update_layout(
                    xaxis_title="P&L ($)",
                    barmode="stack",
                    height=max(300, len(df_t) * 22),
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        font=dict(size=10),
                    ),
                    margin=dict(l=60, r=20, t=40, b=40),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # Detail table
                display_df = df_t[["ticker", "sector", "pnl", "final_shares",
                                   "final_price", "final_value"]].copy()
                display_df.columns = [
                    "Ticker", "Sector", "P&L ($)", "Shares",
                    "Price ($)", "Value ($)",
                ]
                display_df = display_df.sort_values("Ticker")
                st.dataframe(
                    display_df.style.format({
                        "P&L ($)": "${:,.2f}",
                        "Shares": "{:,.1f}",
                        "Price ($)": "${:,.2f}",
                        "Value ($)": "${:,.2f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                    height=min(400, len(display_df) * 36 + 40),
                )


# ===== Tab 4: Normalized to Treasuries ========================================
with tab4:
    st.header("Portfolio Value vs Treasuries")
    st.caption("Values above 1.0 = beating risk-free Treasuries")

    if not runs_with_data:
        st.info("No selected runs have daily time series data.")
    else:
        fig_treas = go.Figure()
        for i, rid in enumerate(runs_with_data):
            run = store.get_run(rid)
            rf_rate = run.get("rf_rate", 0.0) or 0.0
            rows = daily_multi[rid]
            dates = [r["date"] for r in rows]
            values = np.array([r["portfolio_value"] for r in rows])
            initial = values[0]

            # Treasury growth curve
            day0 = pd.Timestamp(dates[0])
            days_elapsed = np.array(
                [(pd.Timestamp(d) - day0).days for d in dates]
            )
            treasury = initial * (1 + rf_rate) ** (days_elapsed / 365.25)
            normalized = values / treasury

            color = TOL_BRIGHT[i % len(TOL_BRIGHT)]
            fig_treas.add_trace(go.Scatter(
                x=dates, y=normalized.tolist(), mode="lines",
                name=f"#{rid} {run['strategy_name']}",
                line=dict(color=color, width=2),
            ))

        fig_treas.add_hline(
            y=1.0, line_dash="dash", line_color="black",
            opacity=0.5, annotation_text="= Treasuries",
        )
        fig_treas.update_layout(
            yaxis_title="Portfolio Value / Treasury Growth",
            xaxis_title="Date",
            hovermode="x unified",
            height=500,
        )
        st.plotly_chart(fig_treas, use_container_width=True)


# ===== Tab 5: Normalized to SPY ===============================================
with tab5:
    st.header("Portfolio Value vs SPY Buy-and-Hold")
    st.caption("Values above 1.0 = beating SPY")

    if not runs_with_data:
        st.info("No selected runs have daily time series data.")
    else:
        # Determine date range from selected runs
        all_dates = []
        for rid in runs_with_data:
            rows = daily_multi[rid]
            if rows:
                all_dates.append(rows[0]["date"])
                all_dates.append(rows[-1]["date"])
        if all_dates:
            start_date = min(all_dates)
            end_date = max(all_dates)
        else:
            start_date, end_date = None, None

        if start_date and end_date:
            spy_hist = fetch_spy(start_date, end_date)

            if spy_hist.empty:
                st.warning("Could not fetch SPY data.")
            else:
                spy_close = spy_hist["Close"].values
                spy_timestamps = [
                    pd.Timestamp(d).timestamp() for d in spy_hist.index
                ]

                fig_spy = go.Figure()
                for i, rid in enumerate(runs_with_data):
                    run = store.get_run(rid)
                    initial_cash = run.get("initial_cash", 0) or 0
                    if initial_cash <= 0:
                        continue
                    rows = daily_multi[rid]
                    dates = [r["date"] for r in rows]
                    values = np.array([r["portfolio_value"] for r in rows])

                    # SPY buy-and-hold growth from initial cash
                    spy_value = initial_cash * (spy_close / spy_close[0])

                    # Align SPY to strategy dates via interpolation
                    strat_ts = [pd.Timestamp(d).timestamp() for d in dates]
                    spy_aligned = np.interp(strat_ts, spy_timestamps, spy_value)

                    normalized = values / spy_aligned
                    color = TOL_BRIGHT[i % len(TOL_BRIGHT)]
                    fig_spy.add_trace(go.Scatter(
                        x=dates, y=normalized.tolist(), mode="lines",
                        name=f"#{rid} {run['strategy_name']}",
                        line=dict(color=color, width=2),
                    ))

                fig_spy.add_hline(
                    y=1.0, line_dash="dash", line_color="black",
                    opacity=0.5, annotation_text="= SPY",
                )
                fig_spy.update_layout(
                    yaxis_title="Portfolio Value / SPY Buy-and-Hold",
                    xaxis_title="Date",
                    hovermode="x unified",
                    height=500,
                )
                st.plotly_chart(fig_spy, use_container_width=True)
        else:
            st.info("Could not determine date range from selected runs.")
