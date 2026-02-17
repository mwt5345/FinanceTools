"""
SQLite storage for backtest results.

Provides persistent, queryable storage so results survive across runs
and can be compared side-by-side without re-executing backtests.

DB file: backtest_results.db (gitignored)

Usage:
    from finance_tools.data.results import ResultsStore
    store = ResultsStore()          # default DB path
    run_id = store.save(result)     # PortfolioBacktestResult
    store.list_runs()
    store.compare([1, 2, 3])
"""

import json
import os
import sqlite3
from datetime import datetime
from typing import Optional

# Default DB location: same directory as this file
_DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "backtest_results.db")


class ResultsStore:
    """SQLite-backed store for backtest results."""

    def __init__(self, db_path: str = _DEFAULT_DB):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT NOT NULL,
                strategy_name   TEXT NOT NULL,
                tickers         TEXT NOT NULL,
                n_tickers       INTEGER NOT NULL,
                period_start    TEXT,
                period_end      TEXT,
                n_trading_days  INTEGER,
                initial_cash    REAL,
                total_contributed REAL,
                monthly_contribution REAL DEFAULT 0.0,
                cash_reserve_pct REAL,
                final_value     REAL,
                total_return    REAL,
                annualized_return REAL,
                annualized_volatility REAL,
                sharpe_ratio    REAL,
                max_drawdown    REAL,
                rf_rate         REAL DEFAULT 0.0,
                n_trades        INTEGER,
                n_dividends     INTEGER,
                final_cash      REAL,
                notes           TEXT,
                runner_script   TEXT
            );

            CREATE TABLE IF NOT EXISTS ticker_contributions (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id   INTEGER NOT NULL,
                ticker   TEXT NOT NULL,
                sector   TEXT,
                pnl      REAL,
                final_shares REAL,
                final_price  REAL,
                final_value  REAL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS daily_values (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          INTEGER NOT NULL,
                date            TEXT NOT NULL,
                portfolio_value REAL NOT NULL,
                cash_value      REAL NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_daily_values_run_id
                ON daily_values(run_id);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, result, notes: str | None = None,
             runner_script: str | None = None) -> int:
        """Save a PortfolioBacktestResult. Returns run_id."""
        from finance_tools.data.universe import get_sector

        tickers = result.tickers
        dv = result.daily_values

        # Period bounds
        period_start = str(dv.index[0].date()) if len(dv) > 0 else None
        period_end = str(dv.index[-1].date()) if len(dv) > 0 else None

        # Monthly contribution (infer from daily_contributions if available)
        monthly = 0.0
        if hasattr(result, "daily_contributions") and result.daily_contributions is not None:
            nonzero = result.daily_contributions[result.daily_contributions > 0]
            if len(nonzero) > 0:
                monthly = float(nonzero.iloc[0])

        # Final cash
        final_cash = float(result.daily_cash.iloc[-1]) if len(result.daily_cash) > 0 else 0.0

        cur = self._conn.execute("""
            INSERT INTO runs (
                timestamp, strategy_name, tickers, n_tickers,
                period_start, period_end, n_trading_days,
                initial_cash, total_contributed, monthly_contribution,
                cash_reserve_pct, final_value, total_return,
                annualized_return, annualized_volatility, sharpe_ratio,
                max_drawdown, rf_rate, n_trades, n_dividends,
                final_cash, notes, runner_script
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            result.strategy_name,
            json.dumps(tickers),
            len(tickers),
            period_start,
            period_end,
            len(dv),
            result.initial_cash,
            getattr(result, "total_contributed", result.initial_cash),
            monthly,
            None,  # cash_reserve_pct not stored on result
            result.final_value,
            result.total_return,
            result.annualized_return,
            result.annualized_volatility,
            result.sharpe_ratio,
            result.max_drawdown,
            result.rf_rate,
            result.n_trades,
            result.n_dividends,
            final_cash,
            notes,
            runner_script,
        ))
        run_id = cur.lastrowid

        # Ticker contributions
        contributions = result.ticker_contribution()
        for t in tickers:
            final_shares = float(result.daily_positions[t].iloc[-1]) if len(result.daily_positions[t]) > 0 else 0.0
            final_price = float(result.daily_prices[t].iloc[-1]) if len(result.daily_prices[t]) > 0 else 0.0
            self._conn.execute("""
                INSERT INTO ticker_contributions
                    (run_id, ticker, sector, pnl, final_shares, final_price, final_value)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                t,
                get_sector(t),
                contributions.get(t, 0.0),
                final_shares,
                final_price,
                final_shares * final_price,
            ))

        # Daily time series
        self._save_daily_values(run_id, dv, result.daily_cash)

        self._conn.commit()
        return run_id

    def save_single(self, result, notes: str | None = None,
                    runner_script: str | None = None) -> int:
        """Save a single-stock BacktestResult. Returns run_id."""
        from finance_tools.data.universe import get_sector

        dv = result.daily_values
        period_start = str(dv.index[0].date()) if len(dv) > 0 else None
        period_end = str(dv.index[-1].date()) if len(dv) > 0 else None
        final_cash = float(result.daily_cash.iloc[-1]) if len(result.daily_cash) > 0 else 0.0

        cur = self._conn.execute("""
            INSERT INTO runs (
                timestamp, strategy_name, tickers, n_tickers,
                period_start, period_end, n_trading_days,
                initial_cash, total_contributed, monthly_contribution,
                cash_reserve_pct, final_value, total_return,
                annualized_return, annualized_volatility, sharpe_ratio,
                max_drawdown, rf_rate, n_trades, n_dividends,
                final_cash, notes, runner_script
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            result.strategy_name,
            json.dumps([result.strategy_name]),
            1,
            period_start,
            period_end,
            len(dv),
            result.initial_cash,
            result.initial_cash,
            0.0,
            None,
            result.final_value,
            result.total_return,
            result.annualized_return,
            result.annualized_volatility,
            result.sharpe_ratio,
            result.max_drawdown,
            result.rf_rate,
            result.n_trades,
            result.n_dividends,
            final_cash,
            notes,
            runner_script,
        ))
        run_id = cur.lastrowid

        # Daily time series
        self._save_daily_values(run_id, dv, result.daily_cash)

        self._conn.commit()
        return run_id

    # ------------------------------------------------------------------
    # Daily values helper
    # ------------------------------------------------------------------

    def _save_daily_values(self, run_id: int, daily_values, daily_cash):
        """Bulk-insert daily portfolio value and cash series."""
        if daily_values is None or len(daily_values) == 0:
            return
        rows = []
        for i in range(len(daily_values)):
            dt = daily_values.index[i]
            date_str = str(dt.date()) if hasattr(dt, "date") else str(dt)
            pv = float(daily_values.iloc[i])
            cv = float(daily_cash.iloc[i]) if daily_cash is not None and i < len(daily_cash) else 0.0
            rows.append((run_id, date_str, pv, cv))
        self._conn.executemany(
            "INSERT INTO daily_values (run_id, date, portfolio_value, cash_value) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def list_runs(self, strategy: str | None = None, limit: int = 20) -> list[dict]:
        """Return recent runs as list of dicts."""
        sql = "SELECT * FROM runs"
        params = []
        if strategy:
            sql += " WHERE strategy_name = ?"
            params.append(strategy)
        sql += " ORDER BY run_id DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: int) -> dict | None:
        """Return full detail for a single run."""
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_ticker_contributions(self, run_id: int) -> list[dict]:
        """Return per-ticker breakdown for a run."""
        rows = self._conn.execute(
            "SELECT * FROM ticker_contributions WHERE run_id = ? ORDER BY ticker",
            (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_daily_values(self, run_id: int) -> list[dict]:
        """Return daily portfolio/cash values for a run."""
        rows = self._conn.execute(
            "SELECT date, portfolio_value, cash_value "
            "FROM daily_values WHERE run_id = ? ORDER BY date",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def has_daily_values(self, run_id: int) -> bool:
        """Check if a run has stored daily values."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM daily_values WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return row["cnt"] > 0

    def get_daily_values_multi(self, run_ids: list[int]) -> dict[int, list[dict]]:
        """Batch-fetch daily values for multiple runs."""
        result = {rid: [] for rid in run_ids}
        if not run_ids:
            return result
        placeholders = ",".join("?" for _ in run_ids)
        rows = self._conn.execute(
            f"SELECT run_id, date, portfolio_value, cash_value "
            f"FROM daily_values WHERE run_id IN ({placeholders}) ORDER BY run_id, date",
            run_ids,
        ).fetchall()
        for r in rows:
            result[r["run_id"]].append({
                "date": r["date"],
                "portfolio_value": r["portfolio_value"],
                "cash_value": r["cash_value"],
            })
        return result

    def compare(self, run_ids: list[int]) -> str:
        """Return side-by-side formatted comparison of runs."""
        runs = []
        for rid in run_ids:
            run = self.get_run(rid)
            if run:
                runs.append(run)
        if not runs:
            return "No matching runs found."

        # Build formatted table
        metrics = [
            ("Run ID",        "run_id",              "d"),
            ("Strategy",      "strategy_name",       "s"),
            ("Tickers",       "n_tickers",           "d"),
            ("Period",        "_period",             "s"),
            ("Days",          "n_trading_days",      ",d"),
            ("Initial $",     "initial_cash",        ",.0f"),
            ("Final $",       "final_value",         ",.0f"),
            ("Total Return",  "total_return",        ".1%"),
            ("Ann. Return",   "annualized_return",   ".1%"),
            ("Ann. Vol",      "annualized_volatility", ".1%"),
            ("Sharpe",        "sharpe_ratio",        ".2f"),
            ("Max DD",        "max_drawdown",        ".1%"),
            ("Trades",        "n_trades",            ",d"),
        ]

        col_width = 16
        lines = []
        header = f"{'Metric':<20s}"
        for run in runs:
            header += f"  {'#' + str(run['run_id']):>{col_width}s}"
        lines.append(header)
        lines.append("=" * len(header))

        for label, key, fmt in metrics:
            row = f"{label:<20s}"
            for run in runs:
                if key == "_period":
                    val = f"{run.get('period_start', '?')} to {run.get('period_end', '?')}"
                    row += f"  {val:>{col_width}s}"
                else:
                    val = run.get(key)
                    if val is None:
                        row += f"  {'N/A':>{col_width}s}"
                    else:
                        row += f"  {format(val, fmt):>{col_width}s}"
            lines.append(row)

        return "\n".join(lines)

    def best_by(self, metric: str, n: int = 5) -> list[dict]:
        """Return top N runs sorted by metric (descending).

        For max_drawdown, sorts ascending (least negative = best).
        """
        direction = "ASC" if metric == "max_drawdown" else "DESC"
        rows = self._conn.execute(
            f"SELECT * FROM runs WHERE {metric} IS NOT NULL "
            f"ORDER BY {metric} {direction} LIMIT ?",
            (n,)
        ).fetchall()
        return [dict(r) for r in rows]

    def clear_by_script(self, runner_script: str) -> int:
        """Delete all runs from a given runner script. Returns count deleted."""
        rows = self._conn.execute(
            "SELECT run_id FROM runs WHERE runner_script = ?",
            (runner_script,)
        ).fetchall()
        for row in rows:
            self.delete(row["run_id"])
        # Reset autoincrement if table is now empty so IDs start from 1
        remaining = self._conn.execute("SELECT COUNT(*) AS cnt FROM runs").fetchone()
        if remaining["cnt"] == 0:
            self._conn.execute("DELETE FROM sqlite_sequence WHERE name='runs'")
            self._conn.commit()
        return len(rows)

    def delete(self, run_id: int) -> bool:
        """Delete a run and its ticker contributions + daily values. Returns True if found."""
        self._conn.execute(
            "DELETE FROM ticker_contributions WHERE run_id = ?", (run_id,))
        self._conn.execute(
            "DELETE FROM daily_values WHERE run_id = ?", (run_id,))
        cur = self._conn.execute(
            "DELETE FROM runs WHERE run_id = ?", (run_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
