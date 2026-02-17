"""
Canonical 100-stock S&P 500 universe spanning all 11 GICS sectors.

Provides a standard benchmark universe for backtests and the trading
assistant.  Designed as a barbell: quality compounders (low vol, strong
returns) + high-beta growth names (high vol) to maximise volatility
dispersion for inverse-volatility and other factor strategies.

Usage:
    from finance_tools.data.universe import ALL_TICKERS, SECTORS, SP500_UNIVERSE
    from finance_tools.data.universe import REGIME_25, TRADING_ASSISTANT_10
    from finance_tools.data.universe import tickers_by_sector, get_sector, validate_tickers
"""

# {ticker: GICS_sector} — 100 stocks, 11 sectors
# Quality compounders + high-beta growth for max vol dispersion
SP500_UNIVERSE: dict[str, str] = {
    # Technology (18) — mix of compounders + high-beta semis/software
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",      # high-beta
    "AVGO": "Technology",
    "CRM":  "Technology",
    "ADBE": "Technology",
    "CSCO": "Technology",
    "INTC": "Technology",
    "TXN":  "Technology",
    "IBM":  "Technology",
    "QCOM": "Technology",
    "ORCL": "Technology",
    "AMAT": "Technology",      # high-beta semi equipment
    "ADP":  "Technology",
    "AMD":  "Technology",      # high-beta
    "NOW":  "Technology",      # high-beta growth SaaS
    "SNPS": "Technology",      # high-beta semi design
    "CDNS": "Technology",      # high-beta semi design
    # Financials (10)
    "JPM":  "Financials",
    "GS":   "Financials",
    "BLK":  "Financials",
    "BAC":  "Financials",
    "MS":   "Financials",
    "SCHW": "Financials",
    "AFL":  "Financials",
    "CB":   "Financials",
    "AXP":  "Financials",      # replaced MMC (not tradeable on Alpaca)
    "PYPL": "Financials",      # high-beta fintech
    # Healthcare (10)
    "JNJ":  "Healthcare",
    "UNH":  "Healthcare",
    "PFE":  "Healthcare",
    "ABT":  "Healthcare",
    "TMO":  "Healthcare",
    "MRK":  "Healthcare",
    "ABBV": "Healthcare",
    "MDT":  "Healthcare",
    "ISRG": "Healthcare",      # high-beta surgical robotics
    "AMGN": "Healthcare",
    # Consumer Discretionary (10) — includes TSLA for vol dispersion
    "AMZN": "Consumer Discretionary",
    "HD":   "Consumer Discretionary",
    "NKE":  "Consumer Discretionary",
    "MCD":  "Consumer Discretionary",
    "F":    "Consumer Discretionary",
    "LOW":  "Consumer Discretionary",
    "TGT":  "Consumer Discretionary",
    "GPC":  "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",  # ultra high-beta
    "XYZ":  "Consumer Discretionary",  # Block Inc, high-beta fintech
    # Communication (6) — removed yield traps T, VZ; added growth
    "GOOGL": "Communication",
    "META":  "Communication",   # high-beta
    "DIS":   "Communication",
    "NFLX":  "Communication",   # high-beta
    "TMUS":  "Communication",   # growth telecom
    "EA":    "Communication",   # gaming, moderate vol
    # Consumer Staples (8) — removed slow compounders GIS, KMB
    "PG":   "Consumer Staples",
    "KO":   "Consumer Staples",
    "COST": "Consumer Staples",
    "CL":   "Consumer Staples",
    "PEP":  "Consumer Staples",
    "WMT":  "Consumer Staples",
    "MDLZ": "Consumer Staples",
    "SYY":  "Consumer Staples",
    # Industrials (8) — removed MMM (yield trap)
    "HON":  "Industrials",
    "CAT":  "Industrials",
    "GE":   "Industrials",
    "UNP":  "Industrials",
    "EMR":  "Industrials",
    "ITW":  "Industrials",
    "GD":   "Industrials",
    "WM":   "Industrials",
    # Energy (8) — removed KMI (yield trap), added cyclical vol
    "XOM":  "Energy",
    "CVX":  "Energy",
    "COP":  "Energy",
    "SLB":  "Energy",
    "EOG":  "Energy",
    "PSX":  "Energy",
    "OKE":  "Energy",
    "HAL":  "Energy",          # high-beta oilfield services
    # Utilities (4) — trimmed heavily, low vol + low growth
    "NEE": "Utilities",
    "SO":  "Utilities",
    "DUK": "Utilities",
    "SRE": "Utilities",
    # Materials (8) — removed DD, added high-beta semi/cyber
    "LIN":  "Materials",
    "APD":  "Materials",
    "SHW":  "Materials",
    "ECL":  "Materials",
    "NUE":  "Materials",
    "PPG":  "Materials",
    "FCX":  "Materials",       # high-beta commodity
    "MRVL": "Materials",       # NOTE: Marvell is tech but adds vol dispersion
    # Real Estate (4) — trimmed, removed yield traps VICI, WPC
    "AMT":  "Real Estate",
    "PLD":  "Real Estate",
    "EQIX": "Real Estate",
    "SPG":  "Real Estate",
    # Cybersecurity / high-beta growth (added to existing sectors above)
    "PANW": "Technology",      # high-beta cybersecurity
    "FTNT": "Technology",      # high-beta cybersecurity
    "ON":   "Technology",      # high-beta power semis
    "ENPH": "Technology",      # ultra high-beta solar/energy tech
    "DLR":  "Real Estate",
    "RTX":  "Industrials",
}

# Derived constants
ALL_TICKERS: list[str] = sorted(SP500_UNIVERSE.keys())

SECTORS: dict[str, list[str]] = {}
for _ticker, _sector in SP500_UNIVERSE.items():
    SECTORS.setdefault(_sector, []).append(_ticker)
for _sector in SECTORS:
    SECTORS[_sector].sort()

# Named subsets (preserve backward compatibility)
REGIME_25: list[str] = sorted([
    "AAPL", "MSFT", "NVDA",
    "JPM", "GS", "BLK",
    "JNJ", "PFE", "UNH",
    "AMZN", "HD", "NKE",
    "PG", "KO", "COST",
    "XOM", "CVX",
    "HON", "CAT",
    "GOOGL", "DIS",
    "NEE", "SO",
    "LIN",
    "AMT",
])

TRADING_ASSISTANT_10: list[str] = sorted([
    "MSFT", "F", "JNJ", "CL", "NFLX", "COP", "DUK", "NVDA", "GE", "AMT",
])


# =====================================================================
# Helper functions
# =====================================================================

def tickers_by_sector(sector: str) -> list[str]:
    """Return sorted list of tickers in a GICS sector.

    Raises KeyError if sector is unknown.
    """
    if sector not in SECTORS:
        raise KeyError(f"Unknown sector: {sector!r}. "
                       f"Valid: {sorted(SECTORS.keys())}")
    return list(SECTORS[sector])


def get_sector(ticker: str) -> str | None:
    """Return GICS sector for a ticker, or None if not in universe."""
    return SP500_UNIVERSE.get(ticker)


def validate_tickers(tickers: list[str]) -> list[str]:
    """Return only tickers that are in SP500_UNIVERSE, preserving order."""
    valid = set(SP500_UNIVERSE)
    return [t for t in tickers if t in valid]
