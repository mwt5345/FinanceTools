"""
Intraday mean-reversion strategies.

Two families:

1. **Chebyshev** — Adapts the Chebyshev inequality approach from
   01_stock_explorer/strategies.py for tick-level data.  Chebyshev's
   inequality guarantees P(|X - mu| >= k * sigma) <= 1/k^2 for any
   distribution — no normality assumption needed.

2. **Ornstein-Uhlenbeck (OU)** — The physics model for mean-reverting
   prices.  The OU process dX = theta*(mu - X)*dt + sigma*dW is a
   Langevin equation applied to log-prices.  Parameters (theta, mu,
   sigma) are estimated via OLS on discrete log-price increments:
   dX = a + b*X + eps, then theta = -b/dt, mu = -a/b.
   Signal: s = (X - mu) / sigma_eq where sigma_eq = sigma / sqrt(2*theta).

Usage:
    from finance_tools.strategies.intraday import IntradayChebyshevWithCooldown
    from finance_tools.strategies.intraday import IntradayOUWithCooldown
"""

import math

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Import from shared backtest engine
# ---------------------------------------------------------------------------

from finance_tools.backtest.engine import Strategy, Action, Portfolio


class IntradayChebyshev(Strategy):
    """Intraday Chebyshev mean-reversion strategy.

    Monitors tick-to-tick returns over a rolling window and triggers
    trades when the z-score exceeds +/- k standard deviations.

    Parameters
    ----------
    window : int
        Number of ticks for rolling statistics (default 30, ~5 min at 10s).
    k_threshold : float
        Minimum z-score to trigger a trade (default 2.0).
    max_position_frac : float
        Maximum fraction of portfolio value to hold in stock (default 0.80).
    """

    name = "Intraday Chebyshev (k=1.5)"

    def __init__(self, window: int = 30, k_threshold: float = 2.0,
                 max_position_frac: float = 0.80):
        self.window = window
        self.k_threshold = k_threshold
        self.max_position_frac = max_position_frac
        self.name = f"Intraday Chebyshev (k={k_threshold})"

    def decide(self, day: pd.Series, history: pd.DataFrame,
               portfolio: Portfolio) -> Action:
        """Decide whether to buy, sell, or hold.

        Parameters
        ----------
        day : pd.Series
            Current tick/bar (must have "Close" column).
        history : pd.DataFrame
            All ticks up to and including current (must have "Close" column).
        portfolio : Portfolio
            Current portfolio state.

        Returns
        -------
        Action
        """
        # Need enough history for rolling stats
        if len(history) < self.window + 1:
            return Action.hold()

        # Tick-to-tick returns over the last window+1 closes
        closes = history["Close"].iloc[-(self.window + 1):].values
        returns = np.diff(closes) / closes[:-1]

        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)

        if sigma == 0:
            return Action.hold()

        # Current tick's return
        current_return = returns[-1]

        # Z-score
        z = (current_return - mu) / sigma

        if abs(z) < self.k_threshold:
            return Action.hold()

        # Chebyshev fraction: min(1 - 1/k^2, max_position_frac)
        k = abs(z)
        fraction = min(1.0 - 1.0 / (k ** 2), self.max_position_frac)

        price = day["Close"]

        if z < -self.k_threshold:
            # Rare dip -> buy
            if portfolio.cash <= 0:
                return Action.hold()
            available_cash = portfolio.cash * fraction
            n_shares = math.floor(available_cash / price) if price > 0 else 0
            if n_shares < 1:
                return Action.hold()
            return Action.buy_shares(n_shares)

        elif z > self.k_threshold:
            # Rare spike -> sell
            if portfolio.shares <= 0:
                return Action.hold()
            n_shares = math.floor(portfolio.shares * fraction)
            if n_shares < 1:
                return Action.hold()
            return Action.sell_shares(n_shares)

        return Action.hold()

    def compute_z(self, history: pd.DataFrame) -> float | None:
        """Compute the current z-score (for display purposes).

        Returns None if insufficient history.
        """
        if len(history) < self.window + 1:
            return None

        closes = history["Close"].iloc[-(self.window + 1):].values
        returns = np.diff(closes) / closes[:-1]

        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)

        if sigma == 0:
            return 0.0

        return (returns[-1] - mu) / sigma


class IntradayChebyshevWithCooldown(IntradayChebyshev):
    """Intraday Chebyshev with a cooldown period between trades.

    After a trade is executed, the strategy returns HOLD for the next
    ``cooldown_ticks`` ticks to prevent whipsaw trading.

    Parameters
    ----------
    cooldown_ticks : int
        Number of ticks to wait after a trade (default 6, ~1 min at 10s).
    """

    def __init__(self, window: int = 30, k_threshold: float = 2.0,
                 max_position_frac: float = 0.80, cooldown_ticks: int = 6):
        super().__init__(window, k_threshold, max_position_frac)
        self.cooldown_ticks = cooldown_ticks
        self._ticks_since_trade = cooldown_ticks  # start ready to trade
        self.name = f"Intraday Chebyshev (k={k_threshold}, cd={cooldown_ticks})"

    def decide(self, day: pd.Series, history: pd.DataFrame,
               portfolio: Portfolio) -> Action:
        """Decide with cooldown enforcement."""
        # Always increment cooldown counter
        action = super().decide(day, history, portfolio)

        if self._ticks_since_trade < self.cooldown_ticks:
            self._ticks_since_trade += 1
            return Action.hold()

        # If the base strategy wants to trade, reset cooldown
        if action.action.value != "hold":
            self._ticks_since_trade = 0
            return action

        self._ticks_since_trade += 1
        return Action.hold()

    def reset_cooldown(self):
        """Reset cooldown counter (e.g. when resuming a session)."""
        self._ticks_since_trade = self.cooldown_ticks


# =====================================================================
# Ornstein-Uhlenbeck strategy
# =====================================================================

class IntradayOU(Strategy):
    """Intraday Ornstein-Uhlenbeck mean-reversion strategy.

    Fits an OU process to rolling log-prices via OLS and trades when
    the signal s = (X - mu) / sigma_eq exceeds the entry threshold.

    Position sizing is linear: fraction grows from 0 at ``entry_threshold``
    to ``max_position_frac`` at ``max_threshold``.

    Parameters
    ----------
    window : int
        Number of ticks for OLS fit (default 60, ~10 min at 10s).
    entry_threshold : float
        Minimum |signal| to trigger a trade (default 2.0).
    max_threshold : float
        Signal level at which position fraction = max_position_frac (default 3.0).
    max_position_frac : float
        Maximum fraction of portfolio value to deploy (default 0.80).
    dt : float
        Time step between ticks, in arbitrary units (default 1.0).
    """

    name = "Intraday OU (entry=2.0)"

    def __init__(self, window: int = 60, entry_threshold: float = 2.0,
                 max_threshold: float = 3.0, max_position_frac: float = 0.80,
                 dt: float = 1.0):
        self.window = window
        self.entry_threshold = entry_threshold
        self.max_threshold = max_threshold
        self.max_position_frac = max_position_frac
        self.dt = dt
        self.name = f"Intraday OU (entry={entry_threshold})"

    # -----------------------------------------------------------------
    # OU parameter estimation
    # -----------------------------------------------------------------

    def _fit_ou(self, log_prices: np.ndarray):
        """Estimate OU parameters via OLS on log-price increments.

        Model: dX_i = a + b * X_i + eps_i
        where dX_i = X_{i+1} - X_i, X_i = log_prices[i].

        Returns (theta, mu, sigma_eq) or None if theta <= 0
        (no mean reversion detected).
        """
        X = log_prices[:-1]
        dX = np.diff(log_prices)

        n = len(X)
        if n < 3:
            return None

        # OLS: dX = a + b * X
        X_mean = np.mean(X)
        dX_mean = np.mean(dX)
        Sxx = np.sum((X - X_mean) ** 2)

        if Sxx == 0:
            return None

        b = np.sum((X - X_mean) * (dX - dX_mean)) / Sxx
        a = dX_mean - b * X_mean

        # OU parameters
        theta = -b / self.dt
        if theta <= 0:
            return None

        mu = -a / b

        # Residual variance -> sigma
        residuals = dX - (a + b * X)
        sigma2 = np.sum(residuals ** 2) / (n - 2) if n > 2 else 0.0
        sigma = math.sqrt(max(sigma2, 0.0) / self.dt) if self.dt > 0 else 0.0

        # Equilibrium standard deviation
        sigma_eq = sigma / math.sqrt(2 * theta) if theta > 0 else 0.0

        if sigma_eq <= 0:
            return None

        return (theta, mu, sigma_eq)

    # -----------------------------------------------------------------
    # Strategy interface
    # -----------------------------------------------------------------

    def decide(self, day: pd.Series, history: pd.DataFrame,
               portfolio: Portfolio) -> Action:
        """Decide whether to buy, sell, or hold based on OU signal."""
        if len(history) < self.window + 1:
            return Action.hold()

        closes = history["Close"].iloc[-(self.window + 1):].values

        # Guard against non-positive prices
        if np.any(closes <= 0):
            return Action.hold()

        log_prices = np.log(closes)
        params = self._fit_ou(log_prices)

        if params is None:
            return Action.hold()

        theta, mu, sigma_eq = params

        # Current signal
        current_log_price = log_prices[-1]
        s = (current_log_price - mu) / sigma_eq

        if abs(s) < self.entry_threshold:
            return Action.hold()

        # Linear position sizing: 0 at entry_threshold, max at max_threshold
        t = min((abs(s) - self.entry_threshold) /
                max(self.max_threshold - self.entry_threshold, 1e-8), 1.0)
        fraction = t * self.max_position_frac

        price = day["Close"]

        if s < -self.entry_threshold:
            # Price below equilibrium -> buy
            if portfolio.cash <= 0:
                return Action.hold()
            available_cash = portfolio.cash * fraction
            n_shares = math.floor(available_cash / price) if price > 0 else 0
            if n_shares < 1:
                return Action.hold()
            return Action.buy_shares(n_shares)

        elif s > self.entry_threshold:
            # Price above equilibrium -> sell
            if portfolio.shares <= 0:
                return Action.hold()
            n_shares = math.floor(portfolio.shares * fraction)
            if n_shares < 1:
                return Action.hold()
            return Action.sell_shares(n_shares)

        return Action.hold()

    def compute_z(self, history: pd.DataFrame) -> float | None:
        """Compute the current OU signal s (for display purposes).

        Returns None if insufficient history or no mean reversion.
        """
        if len(history) < self.window + 1:
            return None

        closes = history["Close"].iloc[-(self.window + 1):].values

        if np.any(closes <= 0):
            return None

        log_prices = np.log(closes)
        params = self._fit_ou(log_prices)

        if params is None:
            return None

        theta, mu, sigma_eq = params
        return float((log_prices[-1] - mu) / sigma_eq)


class IntradayOUWithCooldown(IntradayOU):
    """Intraday OU with a cooldown period between trades.

    After a trade is executed, the strategy returns HOLD for the next
    ``cooldown_ticks`` ticks to prevent whipsaw trading.

    Parameters
    ----------
    cooldown_ticks : int
        Number of ticks to wait after a trade (default 6, ~1 min at 10s).
    """

    def __init__(self, window: int = 60, entry_threshold: float = 2.0,
                 max_threshold: float = 3.0, max_position_frac: float = 0.80,
                 dt: float = 1.0, cooldown_ticks: int = 6):
        super().__init__(window, entry_threshold, max_threshold,
                         max_position_frac, dt)
        self.cooldown_ticks = cooldown_ticks
        self._ticks_since_trade = cooldown_ticks  # start ready to trade
        self.name = (f"Intraday OU (entry={entry_threshold}, "
                     f"cd={cooldown_ticks})")

    def decide(self, day: pd.Series, history: pd.DataFrame,
               portfolio: Portfolio) -> Action:
        """Decide with cooldown enforcement."""
        action = super().decide(day, history, portfolio)

        if self._ticks_since_trade < self.cooldown_ticks:
            self._ticks_since_trade += 1
            return Action.hold()

        if action.action.value != "hold":
            self._ticks_since_trade = 0
            return action

        self._ticks_since_trade += 1
        return Action.hold()

    def reset_cooldown(self):
        """Reset cooldown counter (e.g. when resuming a session)."""
        self._ticks_since_trade = self.cooldown_ticks
