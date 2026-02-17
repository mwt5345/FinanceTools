"""
Multi-stock portfolio strategies.

Each strategy subclasses PortfolioStrategy and implements decide().
Strategies see all tickers simultaneously and can make cross-stock decisions.
"""

import numpy as np
import pandas as pd

from finance_tools.backtest.engine import Action
from finance_tools.backtest.portfolio import PortfolioStrategy, PortfolioState
from finance_tools.strategies.equal_weight import (
    compute_target_shares, compute_volatility, needs_rebalance as ew_needs_rebalance,
    compute_inv_vol_target_shares, inv_vol_needs_rebalance,
    compute_garman_klass_volatility,
)


class EqualWeightRebalance(PortfolioStrategy):
    """
    Targets equal allocation across all tickers using whole shares.

    Uses the canonical equal-weight inverse-volatility algorithm from
    ``common/equal_weight.py`` to compute optimal whole-share targets
    (floor + greedy bin-packing with inv-vol tiebreaker), then emits
    ``Action.buy_shares()`` / ``Action.sell_shares()`` so the backtester
    executes exact share counts — matching the trading assistant app.

    Rebalances when any position drifts beyond ``threshold`` from the
    target weight, or when excess cash exceeds 2x a single ticker's
    target weight.
    """

    name = "Equal Weight"

    def __init__(self, threshold: float = 0.05, vol_lookback: int = 60):
        self.threshold = threshold
        self.vol_lookback = vol_lookback

    def decide(self, day: dict[str, pd.Series],
               history: dict[str, pd.DataFrame],
               portfolio: PortfolioState) -> dict[str, Action]:
        tickers = sorted(portfolio.positions.keys())
        n = len(tickers)
        if n == 0:
            return {}

        # Check if rebalance needed (skip on day 1 — always buy)
        total_shares = sum(portfolio.positions.get(t, 0.0) for t in tickers)
        if total_shares > 0 and not ew_needs_rebalance(
            portfolio.positions, portfolio.prices,
            portfolio.cash, threshold=self.threshold,
        ):
            return {}

        # Compute optimal whole-share targets (inv-vol tiebreaker baked in)
        targets = compute_target_shares(
            positions=portfolio.positions,
            prices=portfolio.prices,
            cash=portfolio.cash,
            vol_lookback=self.vol_lookback,
            history=history,
        )

        # Diff current vs target → sell/buy actions
        actions = {}
        for t in tickers:
            current = int(portfolio.positions.get(t, 0))
            target = targets.get(t, 0)
            diff = target - current
            if diff < 0:
                actions[t] = Action.sell_shares(-diff)
            elif diff > 0:
                actions[t] = Action.buy_shares(diff)

        return actions


class InverseVolatilityWeight(PortfolioStrategy):
    """
    Allocates proportionally more dollars to low-volatility tickers.

    Uses inverse realized volatility (60-day daily return std) as portfolio
    weights.  Low-vol stocks get larger positions, high-vol stocks get smaller.
    Falls back to equal weight when history is insufficient.

    Same rebalance mechanics as EqualWeightRebalance (threshold + excess cash)
    but checks drift against inverse-vol target weights rather than 1/n.
    """

    name = "Inverse Volatility"

    def __init__(self, threshold: float = 0.05, vol_lookback: int = 60):
        self.threshold = threshold
        self.vol_lookback = vol_lookback

    def decide(self, day: dict[str, pd.Series],
               history: dict[str, pd.DataFrame],
               portfolio: PortfolioState) -> dict[str, Action]:
        tickers = sorted(portfolio.positions.keys())
        n = len(tickers)
        if n == 0:
            return {}

        # Check if rebalance needed (skip on day 1 — always buy)
        total_shares = sum(portfolio.positions.get(t, 0.0) for t in tickers)
        if total_shares > 0 and not inv_vol_needs_rebalance(
            portfolio.positions, portfolio.prices,
            portfolio.cash, threshold=self.threshold,
            vol_lookback=self.vol_lookback, history=history,
        ):
            return {}

        # Compute optimal whole-share targets weighted by inverse volatility
        targets = compute_inv_vol_target_shares(
            positions=portfolio.positions,
            prices=portfolio.prices,
            cash=portfolio.cash,
            vol_lookback=self.vol_lookback,
            history=history,
        )

        # Diff current vs target → sell/buy actions
        actions = {}
        for t in tickers:
            current = int(portfolio.positions.get(t, 0))
            target = targets.get(t, 0)
            diff = target - current
            if diff < 0:
                actions[t] = Action.sell_shares(-diff)
            elif diff > 0:
                actions[t] = Action.buy_shares(diff)

        return actions


class InverseVolatilityGK(PortfolioStrategy):
    """
    Inverse Volatility variant using Garman-Klass vol estimator.

    Differences from base InverseVolatilityWeight:
    - Garman-Klass OHLC volatility (more efficient estimator than close-to-close)
    - Tighter rebalance threshold (default 3% vs 5%)
    - Lower cash reserve (default 2% vs 5%)
    """

    name = "Inv Vol (Garman-Klass)"

    def __init__(self, threshold: float = 0.03, vol_lookback: int = 60,
                 cash_reserve_pct: float = 0.02):
        self.threshold = threshold
        self.vol_lookback = vol_lookback
        self.cash_reserve_pct = cash_reserve_pct

    def decide(self, day: dict[str, pd.Series],
               history: dict[str, pd.DataFrame],
               portfolio: PortfolioState) -> dict[str, Action]:
        tickers = sorted(portfolio.positions.keys())
        n = len(tickers)
        if n == 0:
            return {}

        # Check if rebalance needed (skip on day 1 — always buy)
        total_shares = sum(portfolio.positions.get(t, 0.0) for t in tickers)
        if total_shares > 0 and not inv_vol_needs_rebalance(
            portfolio.positions, portfolio.prices,
            portfolio.cash, cash_reserve_pct=self.cash_reserve_pct,
            threshold=self.threshold,
            vol_lookback=self.vol_lookback, history=history,
            vol_fn=compute_garman_klass_volatility,
        ):
            return {}

        # Compute optimal whole-share targets weighted by GK inverse volatility
        targets = compute_inv_vol_target_shares(
            positions=portfolio.positions,
            prices=portfolio.prices,
            cash=portfolio.cash,
            cash_reserve_pct=self.cash_reserve_pct,
            vol_lookback=self.vol_lookback,
            history=history,
            vol_fn=compute_garman_klass_volatility,
        )

        # Diff current vs target → sell/buy actions
        actions = {}
        for t in tickers:
            current = int(portfolio.positions.get(t, 0))
            target = targets.get(t, 0)
            diff = target - current
            if diff < 0:
                actions[t] = Action.sell_shares(-diff)
            elif diff > 0:
                actions[t] = Action.buy_shares(diff)

        return actions


class IndependentMeanReversion(PortfolioStrategy):
    """
    Runs Bollinger Band mean reversion on each ticker independently.

    Wraps the same logic as the single-stock MeanReversion strategy,
    demonstrating backward-compatible adapter pattern.
    """

    name = "Independent Mean Rev."

    def __init__(self, window: int = 20, n_std: float = 2.0):
        self.window = window
        self.n_std = n_std
        self.name = f"Independent Mean Rev. (BB {window}/{n_std:.0f})"

    def decide(self, day: dict[str, pd.Series],
               history: dict[str, pd.DataFrame],
               portfolio: PortfolioState) -> dict[str, Action]:
        actions = {}

        for ticker in sorted(portfolio.positions.keys()):
            hist = history[ticker]
            if len(hist) < self.window:
                continue

            closes = hist["Close"].iloc[-self.window:]
            ma = closes.mean()
            std = closes.std()
            upper = ma + self.n_std * std
            lower = ma - self.n_std * std
            price = day[ticker]["Close"]

            alloc = portfolio.allocation(ticker)

            if price < lower and alloc < 0.3:
                actions[ticker] = Action.buy(0.5)
            elif price > upper and portfolio.positions.get(ticker, 0.0) > 0:
                actions[ticker] = Action.sell(0.5)

        return actions


class RelativeStrength(PortfolioStrategy):
    """
    Compares recent returns across all tickers and rotates into
    the strongest performers while selling the weakest.

    Momentum / relative-value cross-stock strategy.
    """

    name = "Relative Strength (20d)"

    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.name = f"Relative Strength ({lookback}d)"

    def decide(self, day: dict[str, pd.Series],
               history: dict[str, pd.DataFrame],
               portfolio: PortfolioState) -> dict[str, Action]:
        tickers = sorted(portfolio.positions.keys())

        # Need enough history for lookback
        if any(len(history[t]) < self.lookback + 1 for t in tickers):
            return {}

        # Compute recent returns for each ticker
        returns = {}
        for t in tickers:
            closes = history[t]["Close"]
            ret = (closes.iloc[-1] - closes.iloc[-self.lookback]) / closes.iloc[-self.lookback]
            returns[t] = ret

        # Rank tickers by return
        ranked = sorted(tickers, key=lambda t: returns[t], reverse=True)

        if len(ranked) < 2:
            return {}

        actions = {}
        n = len(ranked)
        top = ranked[:max(1, n // 3)]      # top third
        bottom = ranked[-max(1, n // 3):]   # bottom third

        # Sell losers
        for t in bottom:
            if portfolio.positions.get(t, 0.0) > 0:
                actions[t] = Action.sell(0.5)

        # Buy winners
        for t in top:
            actions[t] = Action.buy(1.0 / len(top))

        return actions
