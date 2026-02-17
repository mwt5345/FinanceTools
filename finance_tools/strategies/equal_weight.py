"""
Equal-weight inverse-volatility allocation algorithm.

Single source of truth for the trading assistant (app.py + backtest.py)
and the portfolio backtester (EqualWeightRebalance strategy).

The core routine computes optimal whole-share positions via:
  1. Target dollars per ticker = investable / n
  2. Floor to base whole shares
  3. Greedy bin-packing: leftover budget → most-underweight ticker,
     ties broken by inverse volatility (least volatile first)
"""

import math

import pandas as pd

CASH_RESERVE_PCT = 0.05


def compute_volatility(history: dict[str, pd.DataFrame],
                       ticker: str,
                       lookback: int = 60) -> float:
    """Realized volatility (daily return std) for a single ticker.

    Returns ``float('inf')`` when there is insufficient data, so that
    the ticker gets lowest priority in inverse-vol sorting.
    """
    hist = history.get(ticker)
    if hist is None or len(hist) < 2:
        return float("inf")
    closes = hist["Close"].iloc[-lookback:]
    rets = closes.pct_change().dropna()
    if len(rets) < 2:
        return float("inf")
    return float(rets.std())


def compute_garman_klass_volatility(history: dict[str, pd.DataFrame],
                                     ticker: str,
                                     lookback: int = 60) -> float:
    """Garman-Klass volatility estimator using OHLC data.

    Uses the Garman-Klass (1980) formula which incorporates High, Low,
    Open, and Close prices for a more efficient volatility estimate than
    simple close-to-close realized vol.

    GK_var = 0.5 * (log(H/L))^2 - (2*ln(2) - 1) * (log(C/O))^2

    Returns annualized daily std (sqrt of mean daily GK variance).
    Returns ``float('inf')`` when there is insufficient data.
    """
    import numpy as np

    hist = history.get(ticker)
    if hist is None or len(hist) < 2:
        return float("inf")

    df = hist.iloc[-lookback:]
    if len(df) < 2:
        return float("inf")

    for col in ("Open", "High", "Low", "Close"):
        if col not in df.columns:
            return float("inf")

    h = df["High"].values
    l = df["Low"].values
    o = df["Open"].values
    c = df["Close"].values

    # Guard against zero/negative prices
    if np.any(h <= 0) or np.any(l <= 0) or np.any(o <= 0) or np.any(c <= 0):
        return float("inf")

    log_hl = np.log(h / l)
    log_co = np.log(c / o)

    gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    mean_var = float(np.mean(gk_var))

    if mean_var <= 0:
        return float("inf")

    return float(np.sqrt(mean_var))


def compute_target_shares(positions: dict[str, int | float],
                          prices: dict[str, float],
                          cash: float,
                          cash_reserve_pct: float = CASH_RESERVE_PCT,
                          vol_lookback: int = 60,
                          history: dict[str, pd.DataFrame] | None = None,
                          ) -> dict[str, int]:
    """Compute optimal whole-share targets via floor + greedy bin-packing.

    Pure function — no side effects.

    Parameters
    ----------
    positions : {ticker: current_shares}
    prices : {ticker: current_price}
    cash : available cash
    cash_reserve_pct : fraction of total value to keep as cash
    vol_lookback : days of returns for volatility estimate
    history : {ticker: OHLCV DataFrame} for volatility calculation.
              If None, ties are broken alphabetically.

    Returns
    -------
    {ticker: target_whole_shares}
    """
    tickers = sorted(positions.keys())
    n = len(tickers)
    if n == 0:
        return {}

    total_equity = sum(
        positions.get(t, 0) * prices.get(t, 0) for t in tickers
    )
    tv = cash + total_equity
    if tv <= 0:
        return {t: 0 for t in tickers}

    investable = tv * (1 - cash_reserve_pct)
    target_per = investable / n

    # Step 1: floor shares
    target_shares: dict[str, int] = {}
    for t in tickers:
        p = prices.get(t, 0)
        target_shares[t] = math.floor(target_per / p) if p > 0 else 0

    # Step 2: greedy allocation of remaining budget
    base_cost = sum(target_shares[t] * prices.get(t, 0) for t in tickers)
    remaining = investable - base_cost

    if history is None:
        history = {}

    while remaining > 0:
        candidates = [t for t in tickers
                      if prices.get(t, 0) > 0 and prices[t] <= remaining]
        if not candidates:
            break

        def _gap(t):
            return target_per - target_shares[t] * prices[t]

        candidates.sort(key=lambda t: (-_gap(t),
                                        compute_volatility(history, t, vol_lookback)))
        best = candidates[0]
        target_shares[best] += 1
        remaining -= prices[best]

    return target_shares


def compute_inv_vol_target_shares(positions: dict[str, int | float],
                                   prices: dict[str, float],
                                   cash: float,
                                   cash_reserve_pct: float = CASH_RESERVE_PCT,
                                   vol_lookback: int = 60,
                                   history: dict[str, pd.DataFrame] | None = None,
                                   vol_fn=None,
                                   ) -> dict[str, int]:
    """Compute whole-share targets weighted by inverse volatility.

    Like ``compute_target_shares`` but allocates proportionally more dollars
    to low-volatility tickers.  Tickers with infinite volatility (insufficient
    data) receive equal shares of the pool as a fallback.

    Parameters
    ----------
    vol_fn : callable, optional
        Volatility estimator with signature ``(history, ticker, lookback) -> float``.
        Defaults to ``compute_volatility`` (realized close-to-close vol).

    Returns
    -------
    {ticker: target_whole_shares}
    """
    if vol_fn is None:
        vol_fn = compute_volatility

    tickers = sorted(positions.keys())
    n = len(tickers)
    if n == 0:
        return {}

    total_equity = sum(
        positions.get(t, 0) * prices.get(t, 0) for t in tickers
    )
    tv = cash + total_equity
    if tv <= 0:
        return {t: 0 for t in tickers}

    investable = tv * (1 - cash_reserve_pct)

    if history is None:
        history = {}

    # Compute volatilities
    vols = {t: vol_fn(history, t, vol_lookback) for t in tickers}

    # Separate finite-vol tickers from infinite-vol (insufficient data)
    finite = {t: v for t, v in vols.items() if math.isfinite(v) and v > 0}
    infinite = [t for t in tickers if t not in finite]

    # Compute weights: inverse-vol for finite, equal share for infinite
    if finite:
        inv_sum = sum(1.0 / v for v in finite.values())
        weights = {t: (1.0 / v) / inv_sum for t, v in finite.items()}
    else:
        weights = {}

    if infinite:
        # Give infinite-vol tickers an equal fraction of the pool
        inf_weight = 1.0 / n
        for t in infinite:
            weights[t] = inf_weight
        # Re-normalize so all weights sum to 1
        w_sum = sum(weights.values())
        if w_sum > 0:
            weights = {t: w / w_sum for t, w in weights.items()}

    # Dollar targets per ticker
    target_per = {t: weights.get(t, 0) * investable for t in tickers}

    # Step 1: floor shares
    target_shares: dict[str, int] = {}
    for t in tickers:
        p = prices.get(t, 0)
        target_shares[t] = math.floor(target_per[t] / p) if p > 0 else 0

    # Step 2: greedy allocation of remaining budget
    base_cost = sum(target_shares[t] * prices.get(t, 0) for t in tickers)
    remaining = investable - base_cost

    while remaining > 0:
        candidates = [t for t in tickers
                      if prices.get(t, 0) > 0 and prices[t] <= remaining]
        if not candidates:
            break

        def _gap(t):
            return target_per[t] - target_shares[t] * prices[t]

        candidates.sort(key=lambda t: (-_gap(t),
                                        vol_fn(history, t, vol_lookback)))
        best = candidates[0]
        target_shares[best] += 1
        remaining -= prices[best]

    return target_shares


def inv_vol_needs_rebalance(positions: dict[str, int | float],
                            prices: dict[str, float],
                            cash: float,
                            cash_reserve_pct: float = CASH_RESERVE_PCT,
                            threshold: float = 0.05,
                            vol_lookback: int = 60,
                            history: dict[str, pd.DataFrame] | None = None,
                            vol_fn=None,
                            ) -> bool:
    """Check if any ticker drifts beyond *threshold* from its inv-vol target weight."""
    if vol_fn is None:
        vol_fn = compute_volatility

    tickers = sorted(positions.keys())
    n = len(tickers)
    if n == 0:
        return False

    total_equity = sum(
        positions.get(t, 0) * prices.get(t, 0) for t in tickers
    )
    tv = cash + total_equity
    if tv <= 0:
        return False

    if history is None:
        history = {}

    # Compute target weights from current volatilities
    vols = {t: vol_fn(history, t, vol_lookback) for t in tickers}
    finite = {t: v for t, v in vols.items() if math.isfinite(v) and v > 0}
    infinite = [t for t in tickers if t not in finite]

    if finite:
        inv_sum = sum(1.0 / v for v in finite.values())
        weights = {t: (1.0 / v) / inv_sum for t, v in finite.items()}
    else:
        weights = {}

    if infinite:
        inf_weight = 1.0 / n
        for t in infinite:
            weights[t] = inf_weight
        w_sum = sum(weights.values())
        if w_sum > 0:
            weights = {t: w / w_sum for t, w in weights.items()}

    # Scale weights by (1 - cash_reserve_pct) to get portfolio-level targets
    for t in tickers:
        target_weight = weights.get(t, 0) * (1 - cash_reserve_pct)
        current_weight = (positions.get(t, 0) * prices.get(t, 0)) / tv
        if abs(current_weight - target_weight) > threshold:
            return True

    # Excess cash trigger
    cash_weight = cash / tv
    if cash_weight > 1.5 * cash_reserve_pct:
        return True

    return False


def needs_rebalance(positions: dict[str, int | float],
                    prices: dict[str, float],
                    cash: float,
                    cash_reserve_pct: float = CASH_RESERVE_PCT,
                    threshold: float = 0.05) -> bool:
    """Check if any ticker drifts beyond *threshold* from target weight."""
    tickers = sorted(positions.keys())
    n = len(tickers)
    if n == 0:
        return False

    total_equity = sum(
        positions.get(t, 0) * prices.get(t, 0) for t in tickers
    )
    tv = cash + total_equity
    if tv <= 0:
        return False

    target_weight = (1 - cash_reserve_pct) / n
    cash_weight = cash / tv

    for t in tickers:
        current_weight = (positions.get(t, 0) * prices.get(t, 0)) / tv
        if abs(current_weight - target_weight) > threshold:
            return True

    # Excess cash (more than 1.5x the reserve) also triggers
    if cash_weight > 1.5 * cash_reserve_pct:
        return True

    return False


def compute_rebalance_trades(current_positions: dict[str, int | float],
                             target_shares: dict[str, int],
                             prices: dict[str, float],
                             total_value: float,
                             cash_reserve_pct: float = CASH_RESERVE_PCT,
                             rebalance_threshold: float = 0.05,
                             cash: float | None = None,
                             ) -> list[dict]:
    """Diff current vs target, filter by threshold, sells first then buys.

    When *cash* is provided and exceeds ``2 * cash_reserve_pct`` of
    total_value, the per-ticker threshold is bypassed for buys so that
    excess cash gets deployed even when no single ticker drifts enough.

    When *cash* is negative, the per-ticker threshold is bypassed for
    sells so that the portfolio deleverages back to positive cash.

    Returns list of trade dicts with keys:
    action, ticker, shares, price, amount.
    """
    tickers = sorted(current_positions.keys())
    n = len(tickers)
    if n == 0 or total_value <= 0:
        return []

    target_weight = (1 - cash_reserve_pct) / n

    # Excess cash: bypass per-ticker threshold for buys
    excess_cash = False
    # Negative cash: bypass per-ticker threshold for sells (on margin)
    negative_cash = False
    if cash is not None and total_value > 0:
        cash_weight = cash / total_value
        if cash_weight > 1.5 * cash_reserve_pct:
            excess_cash = True
        if cash < 0:
            negative_cash = True

    trades: list[dict] = []

    # Sells first (alphabetical)
    for t in tickers:
        current = int(current_positions.get(t, 0))
        target = target_shares.get(t, 0)
        diff = target - current
        current_weight = (current * prices.get(t, 0)) / total_value if prices.get(t, 0) else 0.0
        if diff < 0 and (negative_cash or abs(current_weight - target_weight) > rebalance_threshold):
            trades.append({
                "action": "SELL",
                "ticker": t,
                "shares": -diff,
                "price": prices[t],
                "amount": -diff * prices[t],
            })

    # Then buys (alphabetical)
    for t in tickers:
        current = int(current_positions.get(t, 0))
        target = target_shares.get(t, 0)
        diff = target - current
        current_weight = (current * prices.get(t, 0)) / total_value if prices.get(t, 0) else 0.0
        if diff > 0 and (excess_cash or abs(current_weight - target_weight) > rebalance_threshold):
            trades.append({
                "action": "BUY",
                "ticker": t,
                "shares": diff,
                "price": prices[t],
                "amount": diff * prices[t],
            })

    return trades


def compute_target_trades(portfolio,
                          history: dict[str, pd.DataFrame],
                          vol_lookback: int = 60,
                          rebalance_threshold: float = 0.05) -> list[dict]:
    """Convenience wrapper preserving the existing call signature.

    Parameters
    ----------
    portfolio : PortfolioState (or any object with .positions, .prices, .cash,
                .total_value() attributes)
    history : {ticker: OHLCV DataFrame}
    vol_lookback : days for volatility estimate
    rebalance_threshold : minimum weight drift to trigger a trade

    Returns
    -------
    List of trade dicts (sells first, then buys).
    """
    target_shares = compute_target_shares(
        positions=portfolio.positions,
        prices=portfolio.prices,
        cash=portfolio.cash,
        cash_reserve_pct=CASH_RESERVE_PCT,
        vol_lookback=vol_lookback,
        history=history,
    )

    return compute_rebalance_trades(
        current_positions=portfolio.positions,
        target_shares=target_shares,
        prices=portfolio.prices,
        total_value=portfolio.total_value(),
        cash_reserve_pct=CASH_RESERVE_PCT,
        rebalance_threshold=rebalance_threshold,
        cash=portfolio.cash,
    )
