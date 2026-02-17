"""
Multi-stock portfolio backtesting engine.

Extends the single-stock backtest framework to handle portfolios of
multiple tickers simultaneously. Strategies see all stocks and can make
cross-stock decisions (e.g., rotate from one sector to another).

Usage:
    from finance_tools.backtest.portfolio import PortfolioStrategy, PortfolioBacktester
    from finance_tools.backtest.engine import Action

    class MyPortfolioStrategy(PortfolioStrategy):
        name = "My Strategy"
        def decide(self, day, history, portfolio):
            return {"AAPL": Action.buy(0.5), "MSFT": Action.sell(1.0)}

    bt = PortfolioBacktester(hist_dict, MyPortfolioStrategy())
    result = bt.run()
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from finance_tools.backtest.engine import Action, ActionType


# =====================================================================
# Portfolio State
# =====================================================================

@dataclass
class PortfolioState:
    """Current portfolio state across all tickers."""
    cash: float
    positions: dict[str, float]   # {ticker: shares}
    prices: dict[str, float]      # {ticker: close_price}

    def equity(self, ticker: str) -> float:
        """Market value of a single position."""
        return self.positions.get(ticker, 0.0) * self.prices.get(ticker, 0.0)

    def total_equity(self) -> float:
        """Total market value of all positions."""
        return sum(self.equity(t) for t in self.positions)

    def total_value(self) -> float:
        """Cash + total equity."""
        return self.cash + self.total_equity()

    def allocation(self, ticker: str) -> float:
        """Fraction of portfolio in a single ticker (0 to 1)."""
        tv = self.total_value()
        if tv == 0:
            return 0.0
        return self.equity(ticker) / tv

    def allocations(self) -> dict[str, float]:
        """Allocation fractions for all tickers."""
        return {t: self.allocation(t) for t in self.positions}


# =====================================================================
# Portfolio Strategy Base Class
# =====================================================================

class PortfolioStrategy:
    """
    Base class for multi-stock portfolio strategies.

    Subclass and implement `decide()`. The engine calls it once per
    trading day with today's data for all tickers, the full history,
    and the current PortfolioState.

    Return a dict of {ticker: Action}. Omit tickers to hold them.
    """

    name: str = "unnamed"

    def decide(self, day: dict[str, pd.Series],
               history: dict[str, pd.DataFrame],
               portfolio: PortfolioState) -> dict[str, Action]:
        raise NotImplementedError


# =====================================================================
# Trade Record
# =====================================================================

@dataclass
class PortfolioTrade:
    """Record of a single trade within the portfolio."""
    date: pd.Timestamp
    ticker: str
    action: str           # "buy", "sell", "dividend"
    shares: float
    price: float
    cash_after: float
    total_value_after: float


# =====================================================================
# Backtest Result
# =====================================================================

@dataclass
class PortfolioBacktestResult:
    """Results from a multi-stock portfolio backtest."""
    strategy_name: str
    initial_cash: float
    final_value: float
    tickers: list[str]
    trades: list[PortfolioTrade]
    daily_values: pd.Series           # total portfolio value
    daily_cash: pd.Series
    daily_positions: dict[str, pd.Series]  # {ticker: daily shares}
    daily_prices: dict[str, pd.Series]     # {ticker: daily close price}
    total_contributed: float = 0.0         # initial + all monthly contributions
    daily_contributions: Optional[pd.Series] = None  # cash injected each day
    rf_rate: float = 0.0
    rf_daily: Optional[pd.Series] = None  # daily annualized rf rate (decimal)

    @property
    def total_return(self) -> float:
        base = self.total_contributed if self.total_contributed > 0 else self.initial_cash
        return (self.final_value / base) - 1

    @property
    def _daily_returns(self) -> pd.Series:
        """Daily holding-period returns, adjusted for cash contributions.

        On contribution days the portfolio jumps by the injected amount.
        Subtract that before computing the return so Sharpe / vol
        reflect market performance only.
        """
        vals = self.daily_values
        if self.daily_contributions is not None and (self.daily_contributions != 0).any():
            adjusted = vals - self.daily_contributions.cumsum()
            prev_adjusted = adjusted.shift(1)
            # On each day: return = (value_after_market - value_before) / value_before
            # value_before = previous day's total value (after its contribution)
            prev_total = vals.shift(1)
            # market_only_value = today's value - today's contribution
            market_only = vals - self.daily_contributions
            returns = (market_only / prev_total - 1).iloc[1:]
        else:
            returns = vals.pct_change().dropna()
        return returns

    @property
    def _avg_rf(self) -> float:
        """Average annualized risk-free rate over the backtest period."""
        if self.rf_daily is not None and len(self.rf_daily) > 0:
            rf_aligned = self.rf_daily.reindex(self.daily_values.index).ffill().bfill().fillna(0.0)
            return float(rf_aligned.mean())
        return self.rf_rate

    @property
    def _daily_excess_returns(self) -> pd.Series:
        """Daily returns minus daily risk-free rate.

        When rf_daily is available, aligns it to portfolio dates via
        forward-fill.  Falls back to constant rf_rate / 252.
        """
        raw = self._daily_returns
        if self.rf_daily is not None and len(self.rf_daily) > 0:
            # Align rf to portfolio return dates, forward-fill gaps
            rf_aligned = self.rf_daily.reindex(raw.index).ffill().bfill().fillna(0.0)
            return raw - rf_aligned / 252
        else:
            return raw - self.rf_rate / 252

    @property
    def annualized_return(self) -> float:
        n_days = len(self.daily_values)
        if n_days <= 1:
            return 0.0
        return (1 + self.total_return) ** (252 / n_days) - 1

    @property
    def annualized_volatility(self) -> float:
        return self._daily_returns.std() * np.sqrt(252)

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

    def ticker_contribution(self) -> dict[str, float]:
        """
        Return attribution by ticker.

        Tracks cash spent buying / received selling for each ticker,
        plus final position value, to compute per-ticker P&L.
        """
        spent = {t: 0.0 for t in self.tickers}
        received = {t: 0.0 for t in self.tickers}

        for trade in self.trades:
            if trade.ticker not in spent:
                continue
            if trade.action == "buy":
                spent[trade.ticker] += trade.shares * trade.price
            elif trade.action == "sell":
                received[trade.ticker] += trade.shares * trade.price

        # Final equity per ticker = final shares * final market price
        result = {}
        for t in self.tickers:
            final_shares = self.daily_positions[t].iloc[-1] if len(self.daily_positions[t]) > 0 else 0.0
            final_price = self.daily_prices[t].iloc[-1] if len(self.daily_prices[t]) > 0 else 0.0
            final_equity = final_shares * final_price
            result[t] = (final_equity + received[t]) - spent[t]
        return result

    def summary(self) -> str:
        total_pct = f"{self.total_return:.1%}"
        ann_pct = f"{self.annualized_return:.1%}"
        vol_pct = f"{self.annualized_volatility:.1%}"
        dd_pct = f"{self.max_drawdown:.1%}"
        lines = [
            f"  Strategy:      {self.strategy_name}",
            f"  Tickers:       {', '.join(self.tickers)}",
            f"  Initial cash:  ${self.initial_cash:,.0f}",
        ]
        if self.total_contributed > self.initial_cash:
            lines.append(f"  Contributed:   ${self.total_contributed:,.0f}")
            net_gain = self.final_value - self.total_contributed
            lines.append(f"  Net gain:      ${net_gain:,.0f}")
        lines += [
            f"  Final value:   ${self.final_value:,.0f}",
            f"  Total return:  {total_pct}",
            f"  Ann. return:   {ann_pct}",
            f"  Ann. vol:      {vol_pct}",
            f"  Sharpe ratio:  {self.sharpe_ratio:.2f} (Rf={self._avg_rf:.1%})",
            f"  Max drawdown:  {dd_pct}",
            f"  Trades:        {self.n_trades}",
            f"  Dividends:     {self.n_dividends}",
        ]
        return "\n".join(lines)


# =====================================================================
# Portfolio Backtester
# =====================================================================

class PortfolioBacktester:
    """
    Runs a PortfolioStrategy over multiple tickers simultaneously.

    Parameters
    ----------
    hist_dict : dict[str, pd.DataFrame]
        {ticker: OHLCV DataFrame}. Will be aligned to common trading dates.
    strategy : PortfolioStrategy
        Multi-stock strategy to evaluate.
    initial_cash : float
        Starting cash (default $10,000).
    cash_reserve_pct : float
        Minimum cash fraction to maintain (default 5%).
    reinvest_dividends : bool
        DRIP per ticker if True (default True).
    monthly_contribution : float
        Cash to inject on the first trading day of each new month (default 0).
    """

    def __init__(self, hist_dict: dict[str, pd.DataFrame],
                 strategy: PortfolioStrategy,
                 initial_cash: float = 10_000.0,
                 cash_reserve_pct: float = 0.05,
                 reinvest_dividends: bool = True,
                 monthly_contribution: float = 0.0):
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.cash_reserve_pct = cash_reserve_pct
        self.reinvest_dividends = reinvest_dividends
        self.monthly_contribution = monthly_contribution

        # Align all DataFrames to common trading dates
        self.tickers = sorted(hist_dict.keys())
        self.hist_dict = self._align(hist_dict)

    def _align(self, hist_dict: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Align all DataFrames to union of dates (outer join).

        Tickers with missing data on a given date get NaN rows, allowing
        partial participation: tickers enter the backtest when their data
        starts and leave when it ends.
        """
        # Normalize to date-only index (handles timezone mismatches)
        normalized = {}
        for ticker, df in hist_dict.items():
            df = df.copy()
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.index = pd.to_datetime(df.index).normalize()
            # Drop duplicate dates (keep first)
            df = df[~df.index.duplicated(keep='first')]
            normalized[ticker] = df

        # Union of all dates
        all_dates: set[pd.Timestamp] = set()
        for df in normalized.values():
            all_dates |= set(df.index)

        self.all_dates = sorted(all_dates)

        aligned = {}
        for ticker, df in normalized.items():
            aligned[ticker] = df.reindex(self.all_dates)

        return aligned

    def run(self) -> PortfolioBacktestResult:
        cash = self.initial_cash
        positions = {t: 0.0 for t in self.tickers}
        trades = []
        daily_values = []
        daily_cash_list = []
        daily_positions = {t: [] for t in self.tickers}
        daily_prices_dict = {t: [] for t in self.tickers}
        daily_contribs = []
        dates = []
        total_contributed = self.initial_cash
        prev_month = None

        n_days = len(self.all_dates)

        for i in range(n_days):
            date = self.all_dates[i]

            # Monthly contribution: inject cash on first trading day of new month
            contribution = 0.0
            current_month = (date.year, date.month)
            if self.monthly_contribution > 0 and prev_month is not None:
                if current_month != prev_month:
                    contribution = self.monthly_contribution
                    cash += contribution
                    total_contributed += contribution
            prev_month = current_month

            # Determine active tickers (non-NaN Close price)
            active_tickers = [
                t for t in self.tickers
                if pd.notna(self.hist_dict[t].iloc[i]["Close"])
            ]

            # If no active tickers, record cash-only day
            if not active_tickers:
                dates.append(date)
                daily_values.append(cash)
                daily_cash_list.append(cash)
                daily_contribs.append(contribution)
                for t in self.tickers:
                    daily_positions[t].append(positions[t])
                    daily_prices_dict[t].append(0.0)
                continue

            # Gather today's data and prices (active tickers only)
            day_dict = {}
            prices = {}
            for t in active_tickers:
                row = self.hist_dict[t].iloc[i]
                day_dict[t] = row
                prices[t] = row["Close"]

            # History up to today (active tickers only, NaN rows dropped)
            history_dict = {}
            for t in active_tickers:
                hist_slice = self.hist_dict[t].iloc[:i + 1]
                history_dict[t] = hist_slice.dropna(subset=["Close"])

            # Dividend reinvestment (DRIP per active ticker)
            if self.reinvest_dividends:
                for t in active_tickers:
                    if positions[t] > 0:
                        div = day_dict[t].get("Dividends", 0.0)
                        if div > 0 and prices[t] > 0:
                            div_cash = positions[t] * div
                            drip_shares = div_cash / prices[t]
                            positions[t] += drip_shares
                            total_eq = sum(
                                positions[tk] * prices[tk]
                                for tk in active_tickers
                            )
                            trades.append(PortfolioTrade(
                                date=date,
                                ticker=t,
                                action="dividend",
                                shares=drip_shares,
                                price=prices[t],
                                cash_after=cash,
                                total_value_after=cash + total_eq,
                            ))

            # Build portfolio state (active tickers only) and get decision
            portfolio = PortfolioState(
                cash=cash,
                positions={t: positions[t] for t in active_tickers},
                prices=dict(prices),
            )
            actions = self.strategy.decide(day_dict, history_dict, portfolio)

            # Execute sells first (frees up cash)
            for t in active_tickers:
                action = actions.get(t)
                if action is None:
                    continue
                if action.action == ActionType.SELL and positions[t] > 0:
                    if action.shares is not None:
                        sell_shares = min(action.shares, positions[t])
                    else:
                        sell_shares = positions[t] * action.fraction
                    cash += sell_shares * prices[t]
                    positions[t] -= sell_shares
                    total_eq = sum(
                        positions[tk] * prices[tk]
                        for tk in active_tickers
                    )
                    trades.append(PortfolioTrade(
                        date=date,
                        ticker=t,
                        action="sell",
                        shares=sell_shares,
                        price=prices[t],
                        cash_after=cash,
                        total_value_after=cash + total_eq,
                    ))

            # Execute buys (with cash reserve enforcement)
            for t in active_tickers:
                action = actions.get(t)
                if action is None:
                    continue
                if action.action == ActionType.BUY and cash > 0:
                    total_eq = sum(
                        positions[tk] * prices[tk]
                        for tk in active_tickers
                    )
                    total_value = cash + total_eq
                    min_cash = total_value * self.cash_reserve_pct
                    available = max(cash - min_cash, 0.0)
                    if action.shares is not None:
                        # Whole-share mode: buy exact count, capped by cash
                        max_affordable = math.floor(available / prices[t]) if prices[t] > 0 else 0
                        buy_count = min(int(action.shares), max_affordable)
                        if buy_count < 1:
                            continue
                        spend = buy_count * prices[t]
                        new_shares = float(buy_count)
                    else:
                        # Fraction mode: buy fraction of available cash
                        spend = available * action.fraction
                        if spend < 0.01:
                            continue
                        new_shares = spend / prices[t]
                    positions[t] += new_shares
                    cash -= spend
                    total_eq = sum(
                        positions[tk] * prices[tk]
                        for tk in active_tickers
                    )
                    trades.append(PortfolioTrade(
                        date=date,
                        ticker=t,
                        action="buy",
                        shares=new_shares,
                        price=prices[t],
                        cash_after=cash,
                        total_value_after=cash + total_eq,
                    ))

            # Record daily state — ALL tickers
            total_eq = sum(
                positions[tk] * prices[tk] for tk in active_tickers
            )
            total = cash + total_eq
            dates.append(date)
            daily_values.append(total)
            daily_cash_list.append(cash)
            daily_contribs.append(contribution)
            for t in self.tickers:
                daily_positions[t].append(positions[t])
                if t in prices:
                    daily_prices_dict[t].append(prices[t])
                else:
                    daily_prices_dict[t].append(0.0)

        return PortfolioBacktestResult(
            strategy_name=self.strategy.name,
            initial_cash=self.initial_cash,
            final_value=daily_values[-1],
            tickers=self.tickers,
            trades=trades,
            daily_values=pd.Series(daily_values, index=dates),
            daily_cash=pd.Series(daily_cash_list, index=dates),
            daily_positions={
                t: pd.Series(daily_positions[t], index=dates)
                for t in self.tickers
            },
            daily_prices={
                t: pd.Series(daily_prices_dict[t], index=dates)
                for t in self.tickers
            },
            total_contributed=total_contributed,
            daily_contributions=pd.Series(daily_contribs, index=dates),
        )
