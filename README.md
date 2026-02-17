# Finance Tools

Backtesting engine, portfolio strategies, and live trading tools for equities.

## Features

- **Backtesting engine** -- single-stock and multi-stock portfolio backtesting with pluggable strategies
- **Portfolio strategies** -- Equal Weight, Inverse Volatility, Mean Reversion (Bollinger Bands), Relative Strength, Regime-Adaptive
- **Monte Carlo stress testing** -- random time-window analysis across strategies
- **Interactive portfolio trader** -- CLI paper-trading advisor with local JSON or Alpaca broker backends
- **Intraday trader** -- Chebyshev z-score and Ornstein-Uhlenbeck mean-reversion strategies, polling or WebSocket streaming
- **Streamlit dashboard** -- interactive backtest comparison and visualization
- **Publication-quality plots** -- SciencePlots + LaTeX rendering, colorblind-safe Tol Bright palette

## Architecture

```
FinanceTools/
├── finance_tools/           # pip-installable library
│   ├── backtest/            # Backtesting engines (single-stock, portfolio, Monte Carlo, intraday)
│   ├── broker/              # Broker abstractions (Alpaca, yfinance data feeds)
│   ├── strategies/          # Trading strategies (equal weight, mean reversion, OU, etc.)
│   ├── data/                # Universe, market data, SQLite results store
│   └── utils/               # Plotting utilities
├── apps/
│   ├── portfolio_trader/    # Interactive CLI (local + Alpaca brokers)
│   ├── intraday_trader/     # Intraday CLI + Streamlit (local + Alpaca)
│   ├── backtester/          # Portfolio backtest runners + stress test
│   └── dashboard/           # Streamlit backtest dashboard
└── tests/                   # 611 pytest tests
```

## Quick Start

```bash
# Install (editable, with all optional deps)
pip install -e ".[all]"

# Run tests
python -m pytest tests/ -v

# Run a portfolio backtest
python apps/backtester/run.py --tickers MSFT AAPL F JNJ --cash 10000

# Launch the portfolio trader (local broker)
python apps/portfolio_trader/app.py --broker local --cash 5000

# Launch the portfolio trader (Alpaca paper trading)
python apps/portfolio_trader/app.py --broker alpaca --profile portfolio

# Launch the intraday trader
python apps/intraday_trader/app.py MSFT --strategy ou --broker local

# Launch the Streamlit dashboard
streamlit run apps/dashboard/app.py
```

## Broker Modes

The portfolio and intraday traders support two broker backends:

| Feature | Local | Alpaca |
|---|---|---|
| State storage | JSON file | Alpaca API (source of truth) |
| Trade execution | Simulated | Real paper orders |
| Dividends | Auto-detected via yfinance | N/A |
| Undo | Batch undo by timestamp | N/A |
| Sharpe tracking | Full replay from transactions | N/A |
| Deposit | Adds to local cash | Shows dashboard link |

## Dependencies

**Required:** numpy, scipy, matplotlib, scienceplots, pandas, yfinance

**Optional:**
- `[alpaca]` -- alpaca-py, pyyaml (for Alpaca paper trading)
- `[streamlit]` -- streamlit, plotly (for dashboards)
- `[dev]` -- pytest

## Tech Stack

Python, pandas, NumPy, SciPy, matplotlib, SciencePlots, yfinance, Alpaca API, Streamlit, SQLite

## License

MIT
