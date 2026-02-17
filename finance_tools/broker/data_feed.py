"""
Pluggable data feed abstraction for intraday price polling and streaming.

Provides a DataFeed ABC and three implementations:
  - YFinanceFeed: polls 1-minute bars from yfinance (max 7 days)
  - AlpacaFeed: polls 1-minute bars from Alpaca (years of history)
  - AlpacaStreamFeed: real-time trade-by-trade WebSocket streaming via Alpaca

Usage:
    from finance_tools.broker.data_feed import YFinanceFeed, AlpacaFeed, AlpacaStreamFeed

    feed = YFinanceFeed("AAPL")
    quote = feed.latest()
    print(quote.price, quote.market_open)

    # Alpaca polling (needs ALPACA_API_KEY / ALPACA_SECRET_KEY env vars)
    feed = AlpacaFeed("AAPL")
    hist = feed.history(lookback_minutes=60*390)  # 60 trading days

    # Alpaca streaming (real-time trades via WebSocket)
    feed = AlpacaStreamFeed("AAPL")
    feed.start()
    tick = feed.get_tick(timeout=5.0)  # blocks until next aggregated tick
    feed.stop()
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, time, timezone, timedelta
import queue
import threading

import pandas as pd
import yfinance as yf


# =====================================================================
# Quote
# =====================================================================

@dataclass
class Quote:
    """A single price snapshot."""
    ticker: str
    price: float
    timestamp: datetime
    bid: float | None = None
    ask: float | None = None
    volume: float = 0.0
    market_open: bool = True


# =====================================================================
# DataFeed ABC
# =====================================================================

class DataFeed(ABC):
    """Abstract base class for price data feeds."""

    @abstractmethod
    def latest(self) -> Quote:
        """Fetch the most recent price quote."""

    @abstractmethod
    def history(self, lookback_minutes: int = 60) -> pd.DataFrame:
        """Fetch recent OHLCV bars.

        Parameters
        ----------
        lookback_minutes : int
            Number of minutes of history to return (best-effort).

        Returns
        -------
        pd.DataFrame with columns: Open, High, Low, Close, Volume
        """

    @abstractmethod
    def is_market_open(self) -> bool:
        """Check if the market is currently open."""


# =====================================================================
# YFinanceFeed
# =====================================================================

# NYSE trading hours in US/Eastern
_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)


def _now_eastern() -> datetime:
    """Current time in US/Eastern (UTC-5 or UTC-4 for DST)."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York"))
    except ImportError:
        # Fallback: assume UTC-5 (EST)
        return datetime.now(timezone(timedelta(hours=-5)))


class YFinanceFeed(DataFeed):
    """Data feed backed by yfinance 1-minute bars.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. "AAPL").
    """

    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self._yf = yf.Ticker(self.ticker)

    def latest(self) -> Quote:
        """Fetch the most recent price.

        During market hours: uses 1-minute bars (``period="1d"``).
        Outside market hours: uses daily close (``period="5d"``),
        with ``market_open=False``.
        """
        market_open = self.is_market_open()

        if market_open:
            hist = self._yf.history(period="1d", interval="1m")
        else:
            hist = pd.DataFrame()

        # Fallback to daily data if 1m bars are empty
        if hist is None or len(hist) == 0:
            hist = self._yf.history(period="5d")
            market_open = False

        if hist is None or len(hist) == 0:
            raise RuntimeError(f"No price data available for {self.ticker}")

        last = hist.iloc[-1]
        ts = hist.index[-1]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()

        return Quote(
            ticker=self.ticker,
            price=float(last["Close"]),
            timestamp=ts,
            bid=None,
            ask=None,
            volume=float(last.get("Volume", 0)),
            market_open=market_open,
        )

    def history(self, lookback_minutes: int = 60) -> pd.DataFrame:
        """Fetch 1-minute bars.

        yfinance provides up to 7 days of 1m data.  We request the
        maximum and trim to ``lookback_minutes``.
        """
        hist = self._yf.history(period="7d", interval="1m")
        if hist is None or len(hist) == 0:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        if lookback_minutes and len(hist) > lookback_minutes:
            hist = hist.iloc[-lookback_minutes:]
        return hist

    def is_market_open(self) -> bool:
        """Check if NYSE is currently open (9:30-16:00 ET, weekdays)."""
        now = _now_eastern()
        # Weekday check (0=Monday, 6=Sunday)
        if now.weekday() >= 5:
            return False
        current_time = now.time()
        return _MARKET_OPEN <= current_time < _MARKET_CLOSE


# =====================================================================
# AlpacaFeed
# =====================================================================

class AlpacaFeed(DataFeed):
    """Data feed backed by the Alpaca Market Data API.

    Provides 1-minute bars with years of history (vs 7 days for yfinance).
    Requires ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. "AAPL").
    api_key : str | None
        Alpaca API key. If None, reads from ALPACA_API_KEY env var.
    secret_key : str | None
        Alpaca secret key. If None, reads from ALPACA_SECRET_KEY env var.
    """

    def __init__(self, ticker: str, api_key: str | None = None,
                 secret_key: str | None = None, use_iex: bool = True):
        import os
        self.ticker = ticker.upper()
        self._api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self._use_iex = use_iex
        if not self._api_key or not self._secret_key:
            raise ValueError(
                "Alpaca credentials required. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY environment variables, or pass them "
                "directly."
            )
        self._client = None  # lazy init

    def _get_client(self):
        """Lazy-initialize the Alpaca client."""
        if self._client is None:
            from alpaca.data import StockHistoricalDataClient
            self._client = StockHistoricalDataClient(
                self._api_key, self._secret_key,
            )
        return self._client

    def latest(self) -> Quote:
        """Fetch the most recent price via Alpaca snapshot."""
        from alpaca.data.enums import DataFeed as AlpacaDataFeed
        from alpaca.data.requests import StockLatestBarRequest

        client = self._get_client()
        market_open = self.is_market_open()

        try:
            feed = AlpacaDataFeed.IEX if self._use_iex else AlpacaDataFeed.SIP
            req = StockLatestBarRequest(
                symbol_or_symbols=[self.ticker], feed=feed)
            bars = client.get_stock_latest_bar(req)
            bar = bars[self.ticker]
            ts = bar.timestamp
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()

            return Quote(
                ticker=self.ticker,
                price=float(bar.close),
                timestamp=ts,
                volume=float(bar.volume),
                market_open=market_open,
            )
        except Exception as e:
            raise RuntimeError(
                f"Alpaca: no price data for {self.ticker}: {e}"
            )

    def history(self, lookback_minutes: int = 60) -> pd.DataFrame:
        """Fetch 1-minute bars from Alpaca.

        Parameters
        ----------
        lookback_minutes : int
            Number of minutes of history. Can be very large (e.g. 60*390
            for ~60 trading days). Alpaca handles pagination internally.

        Returns
        -------
        pd.DataFrame with columns: Open, High, Low, Close, Volume
            Indexed by timezone-aware DatetimeIndex.
        """
        from alpaca.data.enums import DataFeed as AlpacaDataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        client = self._get_client()

        # Convert lookback_minutes to calendar days (trading ~390 min/day)
        trading_days = max(lookback_minutes / 390, 1)
        # Add buffer for weekends/holidays (~1.5x)
        calendar_days = int(trading_days * 1.5) + 2
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=calendar_days)

        feed = AlpacaDataFeed.IEX if self._use_iex else AlpacaDataFeed.SIP
        req = StockBarsRequest(
            symbol_or_symbols=[self.ticker],
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            feed=feed,
        )
        bars = client.get_stock_bars(req)
        df = bars.df

        if df is None or len(df) == 0:
            return pd.DataFrame(
                columns=["Open", "High", "Low", "Close", "Volume"]
            )

        # Alpaca returns a MultiIndex (symbol, timestamp) — drop symbol level
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel("symbol")

        # Rename columns to match yfinance convention
        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        })

        # Keep standard OHLCV columns
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"]
                if c in df.columns]
        df = df[cols]

        # Trim to requested lookback
        if len(df) > lookback_minutes:
            df = df.iloc[-lookback_minutes:]

        return df

    def is_market_open(self) -> bool:
        """Check if NYSE is currently open (9:30-16:00 ET, weekdays)."""
        now = _now_eastern()
        if now.weekday() >= 5:
            return False
        current_time = now.time()
        return _MARKET_OPEN <= current_time < _MARKET_CLOSE


# =====================================================================
# AggregatedTick
# =====================================================================

@dataclass
class AggregatedTick:
    """One aggregated micro-tick built from raw trades.

    Attributes
    ----------
    price : float
        Last trade price in the aggregation window.
    volume : float
        Total volume (sum of all trades in the window).
    high : float
        Highest trade price in the window.
    low : float
        Lowest trade price in the window.
    timestamp : datetime
        Timestamp of the last trade in the window.
    n_trades : int
        Number of raw trades aggregated into this tick.
    """
    price: float
    volume: float
    high: float
    low: float
    timestamp: datetime
    n_trades: int = 1


# =====================================================================
# AlpacaStreamFeed
# =====================================================================

class AlpacaStreamFeed(DataFeed):
    """Real-time trade-by-trade data feed via Alpaca WebSocket.

    Raw trades are aggregated into micro-ticks (default 1 second) to keep
    strategies responsive without thrashing. The WebSocket runs in a
    background daemon thread; the main thread consumes aggregated ticks
    via ``get_tick()`` (blocking) or ``drain_ticks()`` (non-blocking).

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. "AAPL").
    api_key : str | None
        Alpaca API key. If None, reads from ALPACA_API_KEY env var.
    secret_key : str | None
        Alpaca secret key. If None, reads from ALPACA_SECRET_KEY env var.
    agg_interval : float
        Aggregation interval in seconds (default 1.0). Raw trades within
        each interval are merged into a single ``AggregatedTick``.
    use_iex : bool
        If True (default), use the free IEX feed. Otherwise use SIP.
    """

    def __init__(self, ticker: str, api_key: str | None = None,
                 secret_key: str | None = None, agg_interval: float = 1.0,
                 use_iex: bool = True):
        import os
        self.ticker = ticker.upper()
        self._api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self._use_iex = use_iex
        self._agg_interval = agg_interval
        if not self._api_key or not self._secret_key:
            raise ValueError(
                "Alpaca credentials required. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY environment variables, or pass them "
                "directly."
            )
        # REST feed for history() and latest() fallback
        self._rest_feed = AlpacaFeed(
            ticker, api_key=self._api_key, secret_key=self._secret_key,
            use_iex=use_iex,
        )
        # Streaming state
        self._stream = None
        self._thread: threading.Thread | None = None
        self._tick_queue: queue.Queue[AggregatedTick] = queue.Queue()
        self._buffer: list[tuple[float, float, datetime]] = []  # (price, size, ts)
        self._buffer_lock = threading.Lock()
        self._last_flush: float = 0.0
        self._last_tick: AggregatedTick | None = None
        self._lock = threading.Lock()  # protects _last_tick

    def start(self) -> None:
        """Start the WebSocket stream in a background daemon thread."""
        from alpaca.data.enums import DataFeed as AlpacaDataFeed
        from alpaca.data.live import StockDataStream

        feed_enum = AlpacaDataFeed.IEX if self._use_iex else AlpacaDataFeed.SIP
        self._stream = StockDataStream(
            self._api_key, self._secret_key, feed=feed_enum,
        )

        async def _on_trade(trade):
            """Async handler called by alpaca-py for each raw trade."""
            import time as _time
            price = float(trade.price)
            size = float(trade.size)
            ts = trade.timestamp
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()

            now = _time.monotonic()
            with self._buffer_lock:
                self._buffer.append((price, size, ts))
                if self._last_flush == 0.0:
                    self._last_flush = now
                if now - self._last_flush >= self._agg_interval:
                    self._flush_buffer(now)

        self._stream.subscribe_trades(_on_trade, self.ticker)

        def _run():
            self._stream.run()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def _flush_buffer(self, now: float) -> None:
        """Aggregate buffered trades into one tick and enqueue it.

        Must be called while holding ``_buffer_lock``.
        """
        if not self._buffer:
            self._last_flush = now
            return

        prices = [t[0] for t in self._buffer]
        sizes = [t[1] for t in self._buffer]
        timestamps = [t[2] for t in self._buffer]

        tick = AggregatedTick(
            price=prices[-1],           # last trade price
            volume=sum(sizes),
            high=max(prices),
            low=min(prices),
            timestamp=timestamps[-1],   # last trade timestamp
            n_trades=len(self._buffer),
        )

        self._buffer.clear()
        self._last_flush = now

        with self._lock:
            self._last_tick = tick

        self._tick_queue.put(tick)

    def stop(self) -> None:
        """Stop the WebSocket stream and join the background thread."""
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._stream = None
        self._thread = None

    def get_tick(self, timeout: float = 5.0) -> AggregatedTick | None:
        """Block until the next aggregated tick is available.

        Parameters
        ----------
        timeout : float
            Maximum seconds to wait. Returns None on timeout.

        Returns
        -------
        AggregatedTick or None
        """
        try:
            return self._tick_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain_ticks(self) -> list[AggregatedTick]:
        """Return all queued ticks without blocking.

        Useful for Streamlit fragments that process all pending ticks
        at once on each refresh cycle.
        """
        ticks = []
        while True:
            try:
                ticks.append(self._tick_queue.get_nowait())
            except queue.Empty:
                break
        return ticks

    @property
    def is_streaming(self) -> bool:
        """True if the background WebSocket thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def latest(self) -> Quote:
        """Return latest price: stream data if available, else REST fallback."""
        with self._lock:
            tick = self._last_tick
        if tick is not None:
            return Quote(
                ticker=self.ticker,
                price=tick.price,
                timestamp=tick.timestamp,
                volume=tick.volume,
                market_open=self.is_market_open(),
            )
        return self._rest_feed.latest()

    def history(self, lookback_minutes: int = 60) -> pd.DataFrame:
        """Fetch historical 1-minute bars via REST (for warmup)."""
        return self._rest_feed.history(lookback_minutes=lookback_minutes)

    def is_market_open(self) -> bool:
        """Check if NYSE is currently open (9:30-16:00 ET, weekdays)."""
        now = _now_eastern()
        if now.weekday() >= 5:
            return False
        current_time = now.time()
        return _MARKET_OPEN <= current_time < _MARKET_CLOSE
