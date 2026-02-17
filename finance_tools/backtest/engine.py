"""
Backtesting engine with a plug-and-play strategy interface.

Usage:
    from finance_tools.backtest.engine import Strategy, Action, Backtester

    class MyStrategy(Strategy):
        def decide(self, day, history, portfolio):
            return Action.buy(fraction=1.0)

    bt = Backtester(hist_df, MyStrategy(), initial_cash=10000)
    results = bt.run()
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


# =====================================================================
# Action / Portfolio
# =====================================================================

class ActionType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Action:
    """A trading decision returned by a strategy.

    Two modes:
    - **Fraction mode** (default): ``fraction`` of available cash (buy) or
      current shares (sell).  Used by most strategies.
    - **Absolute-share mode**: ``shares`` specifies an exact whole-share
      count, overriding ``fraction``.  Used by strategies that compute
      optimal whole-share targets (e.g. EqualWeightRebalance).
    """
    action: ActionType
    fraction: float = 1.0       # fraction of cash (buy) or shares (sell)
    shares: float | None = None  # absolute share count (overrides fraction)

    @staticmethod
    def buy(fraction: float = 1.0) -> "Action":
        return Action(ActionType.BUY, min(max(fraction, 0.0), 1.0))

    @staticmethod
    def sell(fraction: float = 1.0) -> "Action":
        return Action(ActionType.SELL, min(max(fraction, 0.0), 1.0))

    @staticmethod
    def hold() -> "Action":
        return Action(ActionType.HOLD, 0.0)

    @staticmethod
    def buy_shares(n: int) -> "Action":
        """Buy exactly *n* whole shares (capped by available cash at execution)."""
        return Action(ActionType.BUY, fraction=0.0, shares=float(n))

    @staticmethod
    def sell_shares(n: int) -> "Action":
        """Sell exactly *n* shares (capped at current position at execution)."""
        return Action(ActionType.SELL, fraction=0.0, shares=float(n))


@dataclass
class Portfolio:
    """Current portfolio state, passed to strategy each day."""
    cash: float
    shares: float
    price: float  # current close price

    @property
    def equity(self) -> float:
        return self.shares * self.price

    @property
    def total_value(self) -> float:
        return self.cash + self.equity

    @property
    def allocation(self) -> float:
        """Fraction of portfolio in stock (0 to 1)."""
        if self.total_value == 0:
            return 0.0
        return self.equity / self.total_value


# =====================================================================
# Strategy Base Class
# =====================================================================

class Strategy:
    """
    Base class for trading strategies.

    Subclass and implement `decide()` to create a new strategy.
    The engine calls `decide()` once per trading day with:
        - day: current row from the OHLCV DataFrame
        - history: all rows up to and including today
        - portfolio: current Portfolio state

    Return an Action (buy/sell/hold with a fraction).
    """

    name: str = "unnamed"

    def decide(self, day: pd.Series, history: pd.DataFrame,
               portfolio: Portfolio) -> Action:
        raise NotImplementedError


# =====================================================================
# Backtester
# =====================================================================

@dataclass
class Trade:
    """Record of a single trade."""
    date: pd.Timestamp
    action: str
    shares: float
    price: float
    cash_after: float
    equity_after: float


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    strategy_name: str
    initial_cash: float
    final_value: float
    trades: list
    daily_values: pd.Series      # indexed by date
    daily_cash: pd.Series
    daily_shares: pd.Series
    rf_rate: float = 0.0
    rf_daily: Optional[pd.Series] = None  # daily annualized rf rate (decimal)

    @property
    def _avg_rf(self) -> float:
        """Average annualized risk-free rate over the backtest period."""
        if self.rf_daily is not None and len(self.rf_daily) > 0:
            rf_aligned = self.rf_daily.reindex(self.daily_values.index).ffill().bfill().fillna(0.0)
            return float(rf_aligned.mean())
        return self.rf_rate

    @property
    def _daily_excess_returns(self) -> pd.Series:
        """Daily returns minus daily risk-free rate."""
        raw = self.daily_values.pct_change().dropna()
        if self.rf_daily is not None and len(self.rf_daily) > 0:
            rf_aligned = self.rf_daily.reindex(raw.index).ffill().bfill().fillna(0.0)
            return raw - rf_aligned / 252
        else:
            return raw - self.rf_rate / 252

    @property
    def total_return(self) -> float:
        return (self.final_value / self.initial_cash) - 1

    @property
    def annualized_return(self) -> float:
        n_days = len(self.daily_values)
        if n_days <= 1:
            return 0.0
        return (1 + self.total_return) ** (252 / n_days) - 1

    @property
    def annualized_volatility(self) -> float:
        daily_returns = self.daily_values.pct_change().dropna()
        return daily_returns.std() * np.sqrt(252)

    @property
    def sharpe_ratio(self) -> float:
        excess = self._daily_excess_returns
        if len(excess) < 2:
            return 0.0
        std = excess.std()
        if std == 0:
            return 0.0
        return float(excess.mean() / std * np.sqrt(252))

    @property
    def max_drawdown(self) -> float:
        cummax = self.daily_values.cummax()
        drawdown = (self.daily_values - cummax) / cummax
        return drawdown.min()

    @property
    def n_trades(self) -> int:
        return len([t for t in self.trades
                    if t.action not in ("hold", "dividend")])

    @property
    def n_dividends(self) -> int:
        return len([t for t in self.trades if t.action == "dividend"])

    @property
    def total_dividends_reinvested(self) -> float:
        return sum(t.shares * t.price for t in self.trades
                   if t.action == "dividend")

    def summary(self) -> str:
        total_pct = f"{self.total_return:.1%}"
        ann_pct = f"{self.annualized_return:.1%}"
        vol_pct = f"{self.annualized_volatility:.1%}"
        dd_pct = f"{self.max_drawdown:.1%}"
        lines = [
            f"  Strategy:      {self.strategy_name}",
            f"  Initial cash:  ${self.initial_cash:,.0f}",
            f"  Final value:   ${self.final_value:,.0f}",
            f"  Total return:  {total_pct}",
            f"  Ann. return:   {ann_pct}",
            f"  Ann. vol:      {vol_pct}",
            f"  Sharpe ratio:  {self.sharpe_ratio:.2f} (Rf={self._avg_rf:.1%})",
            f"  Max drawdown:  {dd_pct}",
            f"  Trades:        {self.n_trades}",
            f"  Dividends:     {self.n_dividends} (${self.total_dividends_reinvested:,.0f} reinvested)",
        ]
        return "\n".join(lines)


class Backtester:
    """
    Runs a Strategy over historical OHLCV data.

    Parameters
    ----------
    hist : pd.DataFrame
        Historical data with columns: Open, High, Low, Close, Volume.
        Index should be DatetimeIndex.
    strategy : Strategy
        Trading strategy to evaluate.
    initial_cash : float
        Starting cash (default $10,000).
    """

    def __init__(self, hist: pd.DataFrame, strategy: Strategy,
                 initial_cash: float = 10_000.0,
                 cash_reserve_pct: float = 0.05,
                 reinvest_dividends: bool = True):
        self.hist = hist
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.cash_reserve_pct = cash_reserve_pct
        self.reinvest_dividends = reinvest_dividends

    def run(self) -> BacktestResult:
        cash = self.initial_cash
        shares = 0.0
        trades = []
        daily_values = []
        daily_cash = []
        daily_shares = []
        dates = []

        for i in range(len(self.hist)):
            day = self.hist.iloc[i]
            history = self.hist.iloc[:i + 1]
            price = day["Close"]

            # Dividend reinvestment (DRIP)
            if self.reinvest_dividends and shares > 0:
                div = day.get("Dividends", 0.0)
                if div > 0 and price > 0:
                    div_cash = shares * div
                    drip_shares = div_cash / price
                    shares += drip_shares
                    trades.append(Trade(
                        date=day.name, action="dividend",
                        shares=drip_shares, price=price,
                        cash_after=cash,
                        equity_after=shares * price,
                    ))

            portfolio = Portfolio(cash=cash, shares=shares, price=price)
            action = self.strategy.decide(day, history, portfolio)

            # Execute action (enforce cash reserve on buys)
            if action.action == ActionType.BUY and cash > 0:
                total_value = cash + shares * price
                min_cash = total_value * self.cash_reserve_pct
                available = max(cash - min_cash, 0.0)
                if action.shares is not None and action.shares > 0:
                    # Absolute-share mode: buy exact share count
                    spend = min(action.shares * price, available)
                    new_shares = spend / price if price > 0 else 0.0
                else:
                    # Fraction mode: buy fraction of available cash
                    spend = available * action.fraction
                    new_shares = spend / price if price > 0 else 0.0
                if spend >= 0.01:  # skip trivial trades
                    shares += new_shares
                    cash -= spend
                    trades.append(Trade(
                        date=day.name, action="buy", shares=new_shares,
                        price=price, cash_after=cash,
                        equity_after=shares * price,
                    ))
            elif action.action == ActionType.SELL and shares > 0:
                if action.shares is not None and action.shares > 0:
                    # Absolute-share mode: sell exact share count
                    sell_shares = min(action.shares, shares)
                else:
                    # Fraction mode: sell fraction of current shares
                    sell_shares = shares * action.fraction
                if sell_shares > 0:
                    cash += sell_shares * price
                    shares -= sell_shares
                    trades.append(Trade(
                        date=day.name, action="sell", shares=sell_shares,
                        price=price, cash_after=cash,
                        equity_after=shares * price,
                    ))

            total = cash + shares * price
            dates.append(day.name)
            daily_values.append(total)
            daily_cash.append(cash)
            daily_shares.append(shares)

        return BacktestResult(
            strategy_name=self.strategy.name,
            initial_cash=self.initial_cash,
            final_value=daily_values[-1],
            trades=trades,
            daily_values=pd.Series(daily_values, index=dates),
            daily_cash=pd.Series(daily_cash, index=dates),
            daily_shares=pd.Series(daily_shares, index=dates),
        )
