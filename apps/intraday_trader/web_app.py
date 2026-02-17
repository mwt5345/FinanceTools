"""
Intraday Trader -- Streamlit Web Dashboard.

Interactive web UI for the intraday mean-reversion strategy.
Reuses all logic from the CLI app (strategy, risk limits, state, data feed).

Supports two broker modes:
  - local (default): paper trading with local JSON state
  - alpaca: real paper-trading via Alpaca API

Usage:
    cd apps/intraday_trader
    streamlit run web_app.py
"""

import importlib.util
import math
import os
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from finance_tools.backtest.engine import Portfolio, Action, ActionType
from finance_tools.broker.data_feed import YFinanceFeed, AlpacaFeed, AlpacaStreamFeed
from finance_tools.strategies.intraday import (
    IntradayChebyshevWithCooldown, IntradayOUWithCooldown,
)
from finance_tools.broker.alpaca import AlpacaBroker

# ---------------------------------------------------------------------------
# Import app.py via importlib to avoid name collision
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_spec = importlib.util.spec_from_file_location(
    "intraday_app", os.path.join(SCRIPT_DIR, "app.py")
)
intraday_app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(intraday_app)

# Re-export from loaded module
create_initial_state = intraday_app.create_initial_state
save_state = intraday_app.save_state
load_state = intraday_app.load_state
check_risk = intraday_app.check_risk
RiskLimits = intraday_app.RiskLimits
compute_session_stats = intraday_app.compute_session_stats
warmup_history = intraday_app.warmup_history
fetch_portfolio_state = intraday_app.fetch_portfolio_state
execute_action = intraday_app.execute_action

# ---------------------------------------------------------------------------
# Tol Bright palette (colorblind-safe, matches plotting.py / dashboard)
# ---------------------------------------------------------------------------
BLUE = "#4477AA"
CYAN_HEX = "#66CCEE"
GREEN = "#228833"
YELLOW = "#CCBB44"
RED = "#EE6677"
PURPLE = "#AA3377"
GREY = "#BBBBBB"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Intraday Trader",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .status-trading {
        background-color: #228833; color: white; padding: 6px 16px;
        border-radius: 4px; font-weight: bold; display: inline-block;
    }
    .status-stopped {
        background-color: #EE6677; color: white; padding: 6px 16px;
        border-radius: 4px; font-weight: bold; display: inline-block;
    }
    .status-warmup {
        background-color: #CCBB44; color: black; padding: 6px 16px;
        border-radius: 4px; font-weight: bold; display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ===================================================================
# Session state initialization
# ===================================================================

def _init_session_state():
    """Initialize all session_state keys with defaults."""
    defaults = {
        "running": False,
        "state": None,           # the trading state dict
        "feed": None,            # YFinanceFeed / AlpacaFeed / AlpacaStreamFeed
        "strategy": None,        # IntradayChebyshevWithCooldown / OU
        "risk": None,            # RiskLimits
        "tick_prices": [],       # all prices (warmup + live)
        "tick_times": [],        # timestamps for tick_prices
        "trade_count": 0,
        "warmup_done": False,
        "last_z": None,
        "error_msg": None,
        "streaming": False,
        "broker_mode": "local",
        "alpaca_broker": None,   # AlpacaBroker instance (alpaca mode)
        "dry_run": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session_state()


# ===================================================================
# Core actions
# ===================================================================

def _start_trading(ticker: str, cash: float, k: float, window: int,
                   interval: int, cooldown: int,
                   strategy_type: str = "chebyshev",
                   entry: float = 2.0, max_threshold: float = 3.0,
                   feed_type: str = "yfinance",
                   streaming: bool = False,
                   broker_mode: str = "local",
                   profile: str = "intraday",
                   dry_run: bool = False):
    """Create feed, strategy, risk limits, warmup, and start trading."""
    st.session_state.broker_mode = broker_mode
    st.session_state.dry_run = dry_run

    state = create_initial_state(
        ticker, k, window, interval, cooldown,
        strategy_type=strategy_type, entry=entry,
        max_threshold=max_threshold, broker_mode=broker_mode,
        cash=cash, profile=profile,
    )

    # Alpaca broker connection
    alpaca_broker = None
    if broker_mode == "alpaca":
        try:
            alpaca_broker = AlpacaBroker(profile=profile)
        except (ValueError, KeyError, FileNotFoundError) as e:
            st.session_state.error_msg = f"Alpaca connection failed: {e}"
            return
    st.session_state.alpaca_broker = alpaca_broker

    # Create feed -- alpaca broker always uses alpaca feed
    actual_feed = feed_type
    if broker_mode == "alpaca":
        actual_feed = "alpaca"

    if streaming and actual_feed == "alpaca":
        feed = AlpacaStreamFeed(ticker)
        feed.start()
    elif actual_feed == "alpaca":
        feed = AlpacaFeed(ticker)
    else:
        feed = YFinanceFeed(ticker)

    if strategy_type == "ou":
        strategy = IntradayOUWithCooldown(
            window=window, entry_threshold=entry,
            max_threshold=max_threshold, cooldown_ticks=cooldown,
        )
    else:
        strategy = IntradayChebyshevWithCooldown(
            window=window, k_threshold=k, cooldown_ticks=cooldown,
        )
    risk = RiskLimits()

    # Warmup
    warmup_prices = warmup_history(feed, window)
    tick_prices = warmup_prices[:] if warmup_prices else []
    tick_times = [""] * len(tick_prices)

    st.session_state.state = state
    st.session_state.feed = feed
    st.session_state.strategy = strategy
    st.session_state.risk = risk
    st.session_state.tick_prices = tick_prices
    st.session_state.tick_times = tick_times
    st.session_state.trade_count = 0
    st.session_state.warmup_done = len(tick_prices) >= window + 1
    st.session_state.running = True
    st.session_state.last_z = None
    st.session_state.error_msg = None
    st.session_state.streaming = streaming


def _stop_trading():
    """Stop the trading loop and clean up streaming feed if active."""
    st.session_state.running = False
    feed = st.session_state.feed
    if feed is not None and hasattr(feed, "stop"):
        try:
            feed.stop()
        except Exception:
            pass
    st.session_state.streaming = False


def _reset_session():
    """Clear all state and start fresh."""
    feed = st.session_state.get("feed")
    if feed is not None and hasattr(feed, "stop"):
        try:
            feed.stop()
        except Exception:
            pass
    for key in ["state", "feed", "strategy", "risk", "alpaca_broker"]:
        st.session_state[key] = None
    st.session_state.tick_prices = []
    st.session_state.tick_times = []
    st.session_state.trade_count = 0
    st.session_state.warmup_done = False
    st.session_state.running = False
    st.session_state.last_z = None
    st.session_state.error_msg = None
    st.session_state.streaming = False
    st.session_state.broker_mode = "local"
    st.session_state.dry_run = False


def _save_session():
    """Save current state to disk."""
    state = st.session_state.state
    if state is not None:
        save_state(state)
        return True
    return False


def _load_session(ticker: str, feed_type: str = "yfinance",
                  streaming: bool = False, broker_mode: str = "local",
                  profile: str = "intraday"):
    """Load a previous session from disk and reconstruct objects."""
    state = load_state(ticker, broker_mode)
    if state is None:
        return False

    st.session_state.broker_mode = broker_mode

    # Alpaca broker connection
    alpaca_broker = None
    if broker_mode == "alpaca":
        try:
            alpaca_broker = AlpacaBroker(profile=profile)
        except (ValueError, KeyError, FileNotFoundError) as e:
            st.session_state.error_msg = f"Alpaca connection failed: {e}"
            return False
    st.session_state.alpaca_broker = alpaca_broker

    params = state["params"]

    actual_feed = feed_type
    if broker_mode == "alpaca":
        actual_feed = "alpaca"

    if streaming and actual_feed == "alpaca":
        feed = AlpacaStreamFeed(ticker)
    elif actual_feed == "alpaca":
        feed = AlpacaFeed(ticker)
    else:
        feed = YFinanceFeed(ticker)

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
    strategy.reset_cooldown()
    risk = RiskLimits()

    # Rebuild tick history from tick_log
    tick_prices = [t["price"] for t in state.get("tick_log", [])]
    tick_times = [t["timestamp"] for t in state.get("tick_log", [])]

    if len(tick_prices) < params["window"] + 1:
        warmup_prices = warmup_history(feed, params["window"])
        if warmup_prices:
            tick_prices = warmup_prices + tick_prices
            tick_times = [""] * len(warmup_prices) + tick_times

    st.session_state.state = state
    st.session_state.feed = feed
    st.session_state.strategy = strategy
    st.session_state.risk = risk
    st.session_state.tick_prices = tick_prices
    st.session_state.tick_times = tick_times
    st.session_state.trade_count = len(state.get("trades", []))
    st.session_state.warmup_done = len(tick_prices) >= params["window"] + 1
    st.session_state.running = False
    st.session_state.last_z = None
    st.session_state.error_msg = None
    st.session_state.streaming = streaming
    return True


def _process_price_local(price: float) -> bool:
    """Process a single price point for local broker mode.

    Decide -> risk -> execute (local state mutation) -> log.
    Returns True if successfully processed.
    """
    state = st.session_state.state
    strategy = st.session_state.strategy
    risk = st.session_state.risk

    now = datetime.now()
    now_str = now.isoformat(timespec="seconds")

    st.session_state.tick_prices.append(price)
    st.session_state.tick_times.append(now_str)

    history_df = pd.DataFrame({"Close": st.session_state.tick_prices})
    day = history_df.iloc[-1]

    portfolio = Portfolio(
        cash=state["cash"],
        shares=state["shares"],
        price=price,
    )

    action = strategy.decide(day, history_df, portfolio)
    action = check_risk(action, portfolio, risk, st.session_state.trade_count)

    z = strategy.compute_z(history_df)
    st.session_state.last_z = z

    window = state["params"]["window"]
    st.session_state.warmup_done = (
        len(st.session_state.tick_prices) >= window + 1)

    # Execute trade locally
    if action.action == ActionType.BUY and action.shares and action.shares > 0:
        n = int(action.shares)
        cost = n * price
        state["shares"] += n
        state["cash"] -= cost
        st.session_state.trade_count += 1
        state["trades"].append({
            "timestamp": now_str,
            "action": "buy",
            "shares": n,
            "price": price,
            "cost": round(cost, 2),
        })
    elif (action.action == ActionType.SELL and action.shares
          and action.shares > 0):
        n = int(action.shares)
        proceeds = n * price
        state["shares"] -= n
        state["cash"] += proceeds
        st.session_state.trade_count += 1
        state["trades"].append({
            "timestamp": now_str,
            "action": "sell",
            "shares": n,
            "price": price,
            "proceeds": round(proceeds, 2),
        })

    total_value = state["cash"] + state["shares"] * price

    state["tick_log"].append({
        "timestamp": now_str,
        "price": price,
        "z_score": z if z is not None else 0.0,
        "portfolio_value": total_value,
        "shares": state["shares"],
        "cash": state["cash"],
    })

    save_state(state)
    return True


def _process_price_alpaca(price: float) -> bool:
    """Process a single price point for Alpaca broker mode.

    Decide -> risk -> execute (via Alpaca) -> log.
    Returns True if successfully processed.
    """
    state = st.session_state.state
    strategy = st.session_state.strategy
    risk = st.session_state.risk
    broker = st.session_state.alpaca_broker
    dry_run = st.session_state.dry_run
    ticker = state["ticker"]

    now = datetime.now()
    now_str = now.isoformat(timespec="seconds")

    st.session_state.tick_prices.append(price)
    st.session_state.tick_times.append(now_str)

    history_df = pd.DataFrame({"Close": st.session_state.tick_prices})
    day = history_df.iloc[-1]

    # Portfolio from Alpaca
    try:
        portfolio = fetch_portfolio_state(broker, ticker, price)
    except Exception as e:
        st.session_state.error_msg = f"Alpaca error: {e}"
        return False

    action = strategy.decide(day, history_df, portfolio)
    action = check_risk(action, portfolio, risk, st.session_state.trade_count)

    z = strategy.compute_z(history_df)
    st.session_state.last_z = z

    window = state["params"]["window"]
    st.session_state.warmup_done = (
        len(st.session_state.tick_prices) >= window + 1)

    # Execute via Alpaca
    if (action.action == ActionType.BUY and action.shares
            and action.shares > 0):
        execute_action(broker, ticker, action, state, dry_run=dry_run)
        st.session_state.trade_count += 1
    elif (action.action == ActionType.SELL and action.shares
          and action.shares > 0):
        execute_action(broker, ticker, action, state, dry_run=dry_run)
        st.session_state.trade_count += 1

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

    save_state(state)
    return True


def _execute_tick():
    """Execute one or more ticks depending on mode (polling vs streaming).

    In polling mode: fetches a single quote via REST.
    In streaming mode: drains all queued aggregated ticks from the WebSocket.
    Returns True if at least one tick was successfully processed.
    """
    state = st.session_state.state
    feed = st.session_state.feed
    strategy = st.session_state.strategy
    broker_mode = st.session_state.broker_mode

    if state is None or feed is None or strategy is None:
        return False

    # Choose the right price processor
    if broker_mode == "alpaca":
        process_fn = _process_price_alpaca
    else:
        process_fn = _process_price_local

    processed = False

    if st.session_state.streaming:
        ticks = feed.drain_ticks()
        for tick in ticks:
            process_fn(tick.price)
            processed = True
    else:
        try:
            quote = feed.latest()
        except Exception as e:
            st.session_state.error_msg = f"Fetch error: {e}"
            return False
        st.session_state.error_msg = None
        process_fn(quote.price)
        processed = True

    return processed


# ===================================================================
# Rendering helpers
# ===================================================================

def _render_sidebar():
    """Render the sidebar controls. Returns parameters dict."""
    st.sidebar.markdown("### Intraday Trader")
    st.sidebar.caption("Mean Reversion Strategies")
    st.sidebar.divider()

    # Broker selector
    broker_mode = st.sidebar.selectbox(
        "Broker",
        options=["local", "alpaca"],
        format_func=lambda x: {
            "local": "Local (paper trading)",
            "alpaca": "Alpaca (paper account)",
        }[x],
    )

    ticker = st.sidebar.text_input("Ticker", value="F").upper()

    # Broker-specific options
    cash = 10_000.0
    profile = "intraday"
    dry_run = False
    if broker_mode == "local":
        cash = st.sidebar.number_input("Starting Cash ($)", value=10000.0,
                                       min_value=100.0, step=500.0)
    else:
        profile = st.sidebar.text_input("Alpaca Profile", value="intraday")
        dry_run = st.sidebar.checkbox("Dry Run (no orders)", value=False)

    feed_type = st.sidebar.selectbox(
        "Data Feed",
        options=["yfinance", "alpaca"],
        index=1 if broker_mode == "alpaca" else 0,
        format_func=lambda x: {
            "yfinance": "yfinance (free, 7d max)",
            "alpaca": "Alpaca (API key required)",
        }[x],
    )

    strategy_type = st.sidebar.selectbox(
        "Strategy",
        options=["chebyshev", "ou"],
        format_func=lambda x: {
            "chebyshev": "Chebyshev",
            "ou": "Ornstein-Uhlenbeck",
        }[x],
    )

    # Streaming toggle (only for Alpaca feed)
    streaming = False
    actual_feed = feed_type
    if broker_mode == "alpaca":
        actual_feed = "alpaca"
    if actual_feed == "alpaca":
        streaming = st.sidebar.checkbox("Enable WebSocket Streaming",
                                        value=False)

    # Conditional parameters based on strategy
    k = 1.5
    entry = 2.0
    max_threshold = 3.0
    if strategy_type == "chebyshev":
        k = st.sidebar.number_input("k Threshold", value=1.5, min_value=1.0,
                                    max_value=5.0, step=0.1, format="%.1f")
    else:
        entry = st.sidebar.number_input("Entry Threshold", value=2.0,
                                        min_value=0.5, max_value=5.0,
                                        step=0.1, format="%.1f")
        max_threshold = st.sidebar.number_input("Max Threshold", value=3.0,
                                                min_value=1.0, max_value=6.0,
                                                step=0.1, format="%.1f")

    window = st.sidebar.number_input("Window (ticks)", value=30,
                                     min_value=5, max_value=200, step=5)
    interval = st.sidebar.number_input("Interval (sec)", value=60,
                                       min_value=10, max_value=300, step=10)
    cooldown = st.sidebar.number_input("Cooldown (ticks)", value=6,
                                       min_value=0, max_value=30, step=1)

    st.sidebar.divider()

    # Action buttons
    col1, col2, col3 = st.sidebar.columns(3)
    is_running = st.session_state.running
    has_state = st.session_state.state is not None

    with col1:
        if st.button("Start", disabled=is_running, use_container_width=True,
                      type="primary"):
            _start_trading(ticker, cash, k, window, interval, cooldown,
                           strategy_type=strategy_type,
                           entry=entry, max_threshold=max_threshold,
                           feed_type=feed_type, streaming=streaming,
                           broker_mode=broker_mode, profile=profile,
                           dry_run=dry_run)
            st.rerun()
    with col2:
        if st.button("Stop", disabled=not is_running, use_container_width=True):
            _stop_trading()
            st.rerun()
    with col3:
        if st.button("Reset", disabled=is_running, use_container_width=True):
            _reset_session()
            st.rerun()

    st.sidebar.divider()

    # Save / Load
    col_s, col_l = st.sidebar.columns(2)
    with col_s:
        if st.button("Save", disabled=not has_state, use_container_width=True):
            if _save_session():
                st.sidebar.success("Saved!")
            else:
                st.sidebar.error("No session to save.")
    with col_l:
        if st.button("Load", disabled=is_running, use_container_width=True):
            if _load_session(ticker, feed_type=feed_type,
                             streaming=streaming, broker_mode=broker_mode,
                             profile=profile):
                st.sidebar.success(f"Loaded {ticker} session!")
                st.rerun()
            else:
                st.sidebar.warning(f"No saved session for {ticker}.")

    # Session info
    state = st.session_state.state
    if state is not None:
        st.sidebar.divider()
        st.sidebar.markdown("**Session Info**")
        n_ticks = len(state.get("tick_log", []))
        n_trades = len(state.get("trades", []))
        start = state.get("session_start", "N/A")
        bmode = state.get("broker_mode", "local")
        st.sidebar.text(f"Started: {start}")
        st.sidebar.text(f"Broker:  {bmode}")
        st.sidebar.text(f"Ticks:   {n_ticks}")
        st.sidebar.text(f"Trades:  {n_trades}")

    return {
        "ticker": ticker, "cash": cash, "k": k,
        "window": window, "interval": interval, "cooldown": cooldown,
        "strategy_type": strategy_type, "entry": entry,
        "max_threshold": max_threshold, "feed_type": feed_type,
        "streaming": streaming, "broker_mode": broker_mode,
        "profile": profile, "dry_run": dry_run,
    }


def _render_status_bar():
    """Render the colored status indicator."""
    state = st.session_state.state
    if state is None:
        st.markdown("Configure parameters and click **Start** to begin trading.")
        return

    if st.session_state.running:
        if not st.session_state.warmup_done:
            st.markdown(
                '<div class="status-warmup">WARMING UP</div>',
                unsafe_allow_html=True,
            )
        else:
            label = "TRADING"
            if st.session_state.dry_run:
                label = "TRADING (DRY RUN)"
            st.markdown(
                f'<div class="status-trading">{label}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="status-stopped">STOPPED</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.error_msg:
        st.error(st.session_state.error_msg)


def _render_metrics():
    """Render the 6 metric cards."""
    state = st.session_state.state
    if state is None:
        return

    tick_log = state.get("tick_log", [])
    last_price = tick_log[-1]["price"] if tick_log else 0.0
    z = st.session_state.last_z
    broker_mode = state.get("broker_mode", "local")

    if broker_mode == "local":
        total_value = state["cash"] + state["shares"] * last_price
        pnl = total_value - state["initial_cash"]
        pnl_pct = (pnl / state["initial_cash"]
                   if state["initial_cash"] > 0 else 0.0)
        shares = state["shares"]
        cash = state["cash"]
    else:
        # Alpaca: use last tick_log entry
        if tick_log:
            total_value = tick_log[-1].get("portfolio_value", 0.0)
            shares = tick_log[-1].get("shares", 0)
            cash = tick_log[-1].get("cash", 0.0)
        else:
            total_value = 0.0
            shares = 0
            cash = 0.0
        initial_value = tick_log[0].get("portfolio_value", 0.0) if tick_log else 0.0
        pnl = total_value - initial_value
        pnl_pct = pnl / initial_value if initial_value > 0 else 0.0

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("Current Price", f"${last_price:.2f}")
    col2.metric("Z-Score", f"{z:+.2f}" if z is not None else "---")
    col3.metric("Shares Held", f"{shares}")
    col4.metric("Cash", f"${cash:,.2f}")
    col5.metric("Portfolio Value", f"${total_value:,.2f}")
    col6.metric(
        "P&L",
        f"${pnl:+,.2f}",
        delta=f"{pnl_pct:+.2%}",
        delta_color="normal",
    )


def _render_price_chart():
    """Render Plotly price chart with buy/sell markers."""
    state = st.session_state.state
    if state is None:
        return

    tick_log = state.get("tick_log", [])
    if len(tick_log) < 2:
        st.info("Waiting for data... (need at least 2 ticks)")
        return

    timestamps = [t["timestamp"] for t in tick_log]
    prices = [t["price"] for t in tick_log]
    trades = state.get("trades", [])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=timestamps, y=prices, mode="lines",
        name="Price", line=dict(color=BLUE, width=2),
    ))

    # Filter out dry-run/error trades for markers
    valid_trades = [t for t in trades
                    if t.get("status") not in ("dry_run", "error")]

    buy_t = [t["timestamp"] for t in valid_trades if t["action"] == "buy"]
    buy_p = [t["price"] for t in valid_trades if t["action"] == "buy"]
    if buy_t:
        fig.add_trace(go.Scatter(
            x=buy_t, y=buy_p, mode="markers",
            name="Buy",
            marker=dict(symbol="triangle-up", size=12, color=GREEN,
                        line=dict(width=1, color="white")),
        ))

    sell_t = [t["timestamp"] for t in valid_trades if t["action"] == "sell"]
    sell_p = [t["price"] for t in valid_trades if t["action"] == "sell"]
    if sell_t:
        fig.add_trace(go.Scatter(
            x=sell_t, y=sell_p, mode="markers",
            name="Sell",
            marker=dict(symbol="triangle-down", size=12, color=RED,
                        line=dict(width=1, color="white")),
        ))

    fig.update_layout(
        yaxis_title=f"{state['ticker']} Price ($)",
        xaxis_title="Time",
        hovermode="x unified",
        height=400,
        margin=dict(l=60, r=20, t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_zscore_chart():
    """Render Plotly z-score chart with threshold bands."""
    state = st.session_state.state
    if state is None:
        return

    tick_log = state.get("tick_log", [])
    if len(tick_log) < 2:
        st.info("Waiting for data...")
        return

    timestamps = [t["timestamp"] for t in tick_log]
    z_scores = [t.get("z_score", 0.0) for t in tick_log]

    strat_type = state["params"].get("strategy", "chebyshev")
    if strat_type == "ou":
        threshold = state["params"].get("entry_threshold", 2.0)
        y_label = "OU Signal (s)"
        thresh_label = "entry"
    else:
        threshold = state["params"]["k_threshold"]
        y_label = "Z-Score"
        thresh_label = "k"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=timestamps + timestamps[::-1],
        y=[threshold] * len(timestamps) + [-threshold] * len(timestamps),
        fill="toself",
        fillcolor="rgba(187, 187, 187, 0.15)",
        line=dict(width=0),
        name=f"{thresh_label} = {threshold}",
        showlegend=True,
    ))

    fig.add_trace(go.Scatter(
        x=timestamps, y=z_scores, mode="lines",
        name=y_label, line=dict(color=BLUE, width=2),
    ))

    fig.add_hline(y=threshold, line_dash="dash", line_color=RED, opacity=0.7,
                  annotation_text=f"+{thresh_label}={threshold}")
    fig.add_hline(y=-threshold, line_dash="dash", line_color=GREEN, opacity=0.7,
                  annotation_text=f"-{thresh_label}={threshold}")
    fig.add_hline(y=0, line_dash="dot", line_color=GREY, opacity=0.5)

    fig.update_layout(
        yaxis_title=y_label,
        xaxis_title="Time",
        hovermode="x unified",
        height=400,
        margin=dict(l=60, r=20, t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_pnl_chart():
    """Render Plotly cumulative P&L area chart."""
    state = st.session_state.state
    if state is None:
        return

    tick_log = state.get("tick_log", [])
    if len(tick_log) < 2:
        st.info("Waiting for data...")
        return

    timestamps = [t["timestamp"] for t in tick_log]
    values = [t.get("portfolio_value", 0) for t in tick_log]
    broker_mode = state.get("broker_mode", "local")

    if broker_mode == "local":
        initial = state.get("initial_cash", values[0])
    else:
        initial = values[0] if values else 0

    pnl = [v - initial for v in values]

    fig = go.Figure()

    pnl_pos = [max(p, 0) for p in pnl]
    fig.add_trace(go.Scatter(
        x=timestamps, y=pnl_pos, mode="lines",
        fill="tozeroy", fillcolor="rgba(34, 136, 51, 0.2)",
        line=dict(width=0), showlegend=False,
    ))

    pnl_neg = [min(p, 0) for p in pnl]
    fig.add_trace(go.Scatter(
        x=timestamps, y=pnl_neg, mode="lines",
        fill="tozeroy", fillcolor="rgba(238, 102, 119, 0.2)",
        line=dict(width=0), showlegend=False,
    ))

    final_color = GREEN if pnl[-1] >= 0 else RED
    fig.add_trace(go.Scatter(
        x=timestamps, y=pnl, mode="lines",
        name="Cumulative P&L",
        line=dict(color=final_color, width=2),
    ))

    fig.add_hline(y=0, line_dash="dot", line_color=GREY, opacity=0.5)

    fig.update_layout(
        yaxis_title="Cumulative P&L ($)",
        xaxis_title="Time",
        hovermode="x unified",
        height=400,
        margin=dict(l=60, r=20, t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_trade_log():
    """Render the trade log table."""
    state = st.session_state.state
    if state is None:
        return

    trades = state.get("trades", [])
    if not trades:
        st.info("No trades yet.")
        return

    rows = []
    for t in trades:
        status = t.get("status", "")
        rows.append({
            "Time": t["timestamp"],
            "Action": t["action"].upper(),
            "Shares": t["shares"],
            "Price ($)": f"${t['price']:.2f}",
            "Cost/Proceeds ($)": f"${t.get('cost', t.get('proceeds', 0)):.2f}",
            "Status": status if status else "filled",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ===================================================================
# Auto-refresh fragment
# ===================================================================

_REFRESH_INTERVAL = timedelta(seconds=2)


@st.fragment(run_every=_REFRESH_INTERVAL)
def _trading_fragment():
    """Auto-refreshing fragment: execute tick then render charts."""
    if st.session_state.running and st.session_state.state is not None:
        _execute_tick()

    _render_status_bar()
    st.markdown("")  # spacer
    _render_metrics()

    st.divider()

    tab_price, tab_zscore, tab_pnl, tab_log = st.tabs([
        "Price Chart", "Z-Score", "P&L", "Trade Log",
    ])

    with tab_price:
        _render_price_chart()
    with tab_zscore:
        _render_zscore_chart()
    with tab_pnl:
        _render_pnl_chart()
    with tab_log:
        _render_trade_log()


# ===================================================================
# Main layout
# ===================================================================

_render_sidebar()
_trading_fragment()
