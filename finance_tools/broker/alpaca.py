"""
Alpaca paper trading broker abstraction.

Wraps alpaca-py's TradingClient into a simple interface for portfolio
management: read positions/cash, submit market orders, poll for fills.

Credentials (checked in order):
  1. Explicit api_key/secret_key arguments
  2. YAML profile from ~/.alpaca/config.yaml (via ``profile`` parameter)
  3. ALPACA_API_KEY / ALPACA_SECRET_KEY environment variables

Usage:
    from finance_tools.broker.alpaca import AlpacaBroker, load_profile

    broker = AlpacaBroker()                       # env vars
    broker = AlpacaBroker(profile="intraday")     # YAML profile
    broker = AlpacaBroker(api_key="PK...", secret_key="SK...")  # explicit
"""

import os
import time
from dataclasses import dataclass


@dataclass
class PositionInfo:
    """Snapshot of a single position from Alpaca."""
    ticker: str
    qty: int
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_pl_pct: float


@dataclass
class OrderResult:
    """Result of an order submission or fill check."""
    order_id: str
    ticker: str
    side: str           # "buy" or "sell"
    qty: int
    status: str         # "filled", "pending", "rejected", etc.
    filled_qty: int
    filled_price: float | None


CONFIG_PATH = os.path.expanduser("~/.alpaca/config.yaml")


def load_profile(profile: str | None = None) -> dict:
    """Load credentials from ~/.alpaca/config.yaml.

    If *profile* is None, uses ``default_profile`` from the config.
    Falls back to environment variables if the config file doesn't exist.

    Returns
    -------
    dict with keys: api_key, secret_key, paper.

    Raises
    ------
    FileNotFoundError
        If the config file doesn't exist and no env vars are set.
    KeyError
        If the requested profile is not found in the config.
    """
    if not os.path.exists(CONFIG_PATH):
        # Fall back to environment variables
        api_key = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        if not api_key or not secret_key:
            raise FileNotFoundError(
                f"Config file {CONFIG_PATH} not found and ALPACA_API_KEY / "
                "ALPACA_SECRET_KEY environment variables are not set."
            )
        return {"api_key": api_key, "secret_key": secret_key, "paper": True}

    import yaml  # lazy import — only needed when config exists

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    profiles = config.get("profiles", {})
    if profile is None:
        profile = config.get("default_profile", "")

    if profile not in profiles:
        available = ", ".join(profiles.keys()) if profiles else "(none)"
        raise KeyError(
            f"Profile '{profile}' not found in {CONFIG_PATH}. "
            f"Available profiles: {available}"
        )

    entry = profiles[profile]
    return {
        "api_key": entry.get("api_key", ""),
        "secret_key": entry.get("secret_key", ""),
        "paper": entry.get("paper", True),
    }


class AlpacaBroker:
    """Thin wrapper around Alpaca's TradingClient for paper trading.

    Parameters
    ----------
    api_key : str | None
        Alpaca API key. If None, resolved via profile or env vars.
    secret_key : str | None
        Alpaca secret key. If None, resolved via profile or env vars.
    paper : bool
        If True (default), use paper trading endpoint.
    profile : str | None
        Named profile from ~/.alpaca/config.yaml.  If None and no
        explicit keys are provided, uses the config's default_profile
        (falling back to env vars if the config file doesn't exist).
    """

    def __init__(self, api_key: str | None = None,
                 secret_key: str | None = None, paper: bool = True,
                 profile: str | None = None):
        if api_key and secret_key:
            # Explicit credentials take priority
            self._api_key = api_key
            self._secret_key = secret_key
            self._paper = paper
        elif profile is not None:
            # Explicit profile name — load from YAML (or fall back to env)
            creds = load_profile(profile)
            self._api_key = creds["api_key"]
            self._secret_key = creds["secret_key"]
            self._paper = creds.get("paper", paper)
        else:
            # No explicit keys or profile — try env vars directly
            self._api_key = os.environ.get("ALPACA_API_KEY", "")
            self._secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
            self._paper = paper

        if not self._api_key or not self._secret_key:
            raise ValueError(
                "Alpaca credentials required. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY environment variables, pass them "
                "directly, or configure ~/.alpaca/config.yaml."
            )
        self._client = None  # lazy init

    def _get_client(self):
        """Lazy-initialize the TradingClient."""
        if self._client is None:
            from alpaca.trading.client import TradingClient
            self._client = TradingClient(
                self._api_key, self._secret_key, paper=self._paper,
            )
        return self._client

    # -----------------------------------------------------------------
    # Account info
    # -----------------------------------------------------------------

    def get_cash(self) -> float:
        """Return cash balance (matches Alpaca dashboard)."""
        client = self._get_client()
        account = client.get_account()
        return float(account.cash)

    def get_equity(self) -> float:
        """Return total account equity."""
        client = self._get_client()
        account = client.get_account()
        return float(account.equity)

    def get_positions(self) -> dict[str, PositionInfo]:
        """Return all open positions as {ticker: PositionInfo}."""
        client = self._get_client()
        positions = client.get_all_positions()
        result = {}
        for pos in positions:
            result[pos.symbol] = PositionInfo(
                ticker=pos.symbol,
                qty=int(pos.qty),
                avg_entry_price=float(pos.avg_entry_price),
                current_price=float(pos.current_price),
                market_value=float(pos.market_value),
                unrealized_pl=float(pos.unrealized_pl),
                unrealized_pl_pct=float(pos.unrealized_plpc),
            )
        return result

    def get_position(self, ticker: str) -> PositionInfo | None:
        """Return position for a single ticker, or None if not held."""
        client = self._get_client()
        try:
            pos = client.get_open_position(ticker.upper())
            return PositionInfo(
                ticker=pos.symbol,
                qty=int(pos.qty),
                avg_entry_price=float(pos.avg_entry_price),
                current_price=float(pos.current_price),
                market_value=float(pos.market_value),
                unrealized_pl=float(pos.unrealized_pl),
                unrealized_pl_pct=float(pos.unrealized_plpc),
            )
        except Exception:
            return None

    def is_market_open(self) -> bool:
        """Check if the market is currently open via Alpaca clock."""
        client = self._get_client()
        clock = client.get_clock()
        return clock.is_open

    # -----------------------------------------------------------------
    # Asset validation
    # -----------------------------------------------------------------

    def is_tradeable(self, ticker: str) -> bool:
        """Check if a ticker is tradeable on Alpaca.

        Queries Alpaca's asset endpoint. Returns False if the asset
        doesn't exist, isn't active, or isn't tradeable.
        Results are cached for the session.
        """
        if not hasattr(self, "_tradeable_cache"):
            self._tradeable_cache: dict[str, bool] = {}

        upper = ticker.upper()
        if upper in self._tradeable_cache:
            return self._tradeable_cache[upper]

        client = self._get_client()
        try:
            asset = client.get_asset(upper)
            tradeable = bool(asset.tradable)
        except Exception:
            tradeable = False

        self._tradeable_cache[upper] = tradeable
        return tradeable

    def filter_tradeable(self, tickers: list[str]) -> tuple[list[str],
                                                             list[str]]:
        """Filter a list of tickers to only those tradeable on Alpaca.

        Returns
        -------
        (tradeable, excluded) — two lists of ticker strings.
        """
        tradeable = []
        excluded = []
        for t in tickers:
            if self.is_tradeable(t):
                tradeable.append(t)
            else:
                excluded.append(t)
        return tradeable, excluded

    # -----------------------------------------------------------------
    # Order execution
    # -----------------------------------------------------------------

    def buy(self, ticker: str, qty: int) -> OrderResult:
        """Submit a market buy order.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol.
        qty : int
            Number of shares to buy.

        Returns
        -------
        OrderResult with initial order status.
        """
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        client = self._get_client()
        req = MarketOrderRequest(
            symbol=ticker.upper(),
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        order = client.submit_order(req)
        return self._order_to_result(order)

    def sell(self, ticker: str, qty: int) -> OrderResult:
        """Submit a market sell order.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol.
        qty : int
            Number of shares to sell.

        Returns
        -------
        OrderResult with initial order status.
        """
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        client = self._get_client()
        req = MarketOrderRequest(
            symbol=ticker.upper(),
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = client.submit_order(req)
        return self._order_to_result(order)

    def wait_for_fill(self, order_id: str, timeout: float = 30.0,
                      poll_interval: float = 0.5) -> OrderResult:
        """Poll until an order is filled or timeout is reached.

        Parameters
        ----------
        order_id : str
            The order ID to monitor.
        timeout : float
            Maximum seconds to wait (default 30).
        poll_interval : float
            Seconds between polls (default 0.5).

        Returns
        -------
        OrderResult with final status. If not filled within timeout,
        status will reflect the last known state (e.g. "pending").
        """
        client = self._get_client()
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            order = client.get_order_by_id(order_id)
            result = self._order_to_result(order)
            if result.status in ("filled", "canceled", "expired", "rejected"):
                return result
            time.sleep(poll_interval)
        # Timeout — return last known state
        order = client.get_order_by_id(order_id)
        return self._order_to_result(order)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _order_to_result(self, order) -> OrderResult:
        """Convert an alpaca Order object to our OrderResult dataclass."""
        filled_qty = int(order.filled_qty) if order.filled_qty else 0
        filled_price = float(order.filled_avg_price) if order.filled_avg_price else None
        return OrderResult(
            order_id=str(order.id),
            ticker=order.symbol,
            side=order.side.value if hasattr(order.side, "value") else str(order.side),
            qty=int(order.qty),
            status=order.status.value if hasattr(order.status, "value") else str(order.status),
            filled_qty=filled_qty,
            filled_price=filled_price,
        )
