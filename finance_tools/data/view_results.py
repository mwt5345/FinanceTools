"""
CLI tool for querying stored backtest results.

Usage:
    python view_results.py                        # list recent runs
    python view_results.py --best sharpe_ratio 10 # top 10 by Sharpe
    python view_results.py --compare 1 3 5        # side-by-side
    python view_results.py --detail 3             # full detail + ticker contributions
    python view_results.py --delete 7             # remove a run
    python view_results.py --strategy "Equal Weight"  # filter by strategy
"""

import argparse
import json
import sys

from finance_tools.data.results import ResultsStore


def _format_run_row(run: dict) -> str:
    """Format a single run as a compact one-line summary."""
    rid = run["run_id"]
    strat = run["strategy_name"][:25]
    n = run["n_tickers"]
    ret = run.get("annualized_return")
    sharpe = run.get("sharpe_ratio")
    dd = run.get("max_drawdown")
    days = run.get("n_trading_days", 0)
    final = run.get("final_value", 0)

    ret_s = f"{ret:.1%}" if ret is not None else "N/A"
    sharpe_s = f"{sharpe:.2f}" if sharpe is not None else "N/A"
    dd_s = f"{dd:.1%}" if dd is not None else "N/A"

    return (f"  {rid:>4d}  {strat:<25s}  {n:>3d} tkrs  "
            f"{days:>5d} days  ${final:>12,.0f}  "
            f"ret={ret_s:>7s}  sharpe={sharpe_s:>6s}  dd={dd_s:>7s}")


def cmd_list(store: ResultsStore, strategy: str | None, limit: int):
    """List recent runs."""
    runs = store.list_runs(strategy=strategy, limit=limit)
    if not runs:
        print("No runs found.")
        return
    print(f"\n  {'ID':>4s}  {'Strategy':<25s}  {'Tkrs':>3s}  "
          f"{'Days':>5s}  {'Final Value':>13s}  "
          f"{'Return':>8s}  {'Sharpe':>7s}  {'MaxDD':>8s}")
    print("  " + "-" * 95)
    for run in runs:
        print(_format_run_row(run))
    print()


def cmd_detail(store: ResultsStore, run_id: int):
    """Show full detail for a run."""
    run = store.get_run(run_id)
    if not run:
        print(f"Run #{run_id} not found.")
        return

    print(f"\n  Run #{run['run_id']}: {run['strategy_name']}")
    print(f"  {'=' * 50}")
    print(f"  Timestamp:       {run['timestamp']}")
    print(f"  Runner:          {run.get('runner_script', 'N/A')}")
    tickers = json.loads(run["tickers"])
    print(f"  Tickers ({run['n_tickers']}):   {', '.join(tickers[:10])}"
          + (f" ..." if len(tickers) > 10 else ""))
    print(f"  Period:          {run.get('period_start', '?')} to {run.get('period_end', '?')}")
    print(f"  Trading days:    {run.get('n_trading_days', 'N/A')}")
    print(f"  Initial cash:    ${run.get('initial_cash', 0):,.0f}")
    tc = run.get("total_contributed", 0) or 0
    if tc > (run.get("initial_cash", 0) or 0):
        print(f"  Total contrib:   ${tc:,.0f}")
    print(f"  Final value:     ${run.get('final_value', 0):,.0f}")
    print(f"  Final cash:      ${run.get('final_cash', 0):,.0f}")

    ret = run.get("total_return")
    ann = run.get("annualized_return")
    vol = run.get("annualized_volatility")
    sharpe = run.get("sharpe_ratio")
    dd = run.get("max_drawdown")
    rf = run.get("rf_rate", 0)

    print(f"  Total return:    {ret:.1%}" if ret is not None else "  Total return:    N/A")
    print(f"  Ann. return:     {ann:.1%}" if ann is not None else "  Ann. return:     N/A")
    print(f"  Ann. volatility: {vol:.1%}" if vol is not None else "  Ann. volatility: N/A")
    print(f"  Sharpe ratio:    {sharpe:.2f} (Rf={rf:.1%})" if sharpe is not None else "  Sharpe ratio:    N/A")
    print(f"  Max drawdown:    {dd:.1%}" if dd is not None else "  Max drawdown:    N/A")
    print(f"  Trades:          {run.get('n_trades', 'N/A')}")
    print(f"  Dividends:       {run.get('n_dividends', 'N/A')}")
    if run.get("notes"):
        print(f"  Notes:           {run['notes']}")

    # Ticker contributions
    contribs = store.get_ticker_contributions(run["run_id"])
    if contribs:
        print(f"\n  {'Ticker':<8s} {'Sector':<25s} {'P&L':>10s} "
              f"{'Shares':>8s} {'Price':>10s} {'Value':>12s}")
        print(f"  {'-' * 77}")
        for c in contribs:
            sector = c.get("sector") or ""
            print(f"  {c['ticker']:<8s} {sector:<25s} "
                  f"${c.get('pnl', 0):>9,.0f} "
                  f"{c.get('final_shares', 0):>8.1f} "
                  f"${c.get('final_price', 0):>9.2f} "
                  f"${c.get('final_value', 0):>11,.0f}")
    print()


def cmd_compare(store: ResultsStore, run_ids: list[int]):
    """Side-by-side comparison."""
    print()
    print(store.compare(run_ids))
    print()


def cmd_best(store: ResultsStore, metric: str, n: int):
    """Top N runs by a metric."""
    valid_metrics = [
        "sharpe_ratio", "annualized_return", "total_return",
        "final_value", "max_drawdown", "annualized_volatility",
    ]
    if metric not in valid_metrics:
        print(f"Invalid metric '{metric}'. Valid: {', '.join(valid_metrics)}")
        return
    runs = store.best_by(metric, n)
    if not runs:
        print("No runs found.")
        return
    direction = "ascending" if metric == "max_drawdown" else "descending"
    print(f"\n  Top {len(runs)} by {metric} ({direction}):")
    print(f"  {'ID':>4s}  {'Strategy':<25s}  {'Tkrs':>3s}  "
          f"{'Days':>5s}  {'Final Value':>13s}  "
          f"{'Return':>8s}  {'Sharpe':>7s}  {'MaxDD':>8s}")
    print("  " + "-" * 95)
    for run in runs:
        print(_format_run_row(run))
    print()


def cmd_delete(store: ResultsStore, run_id: int):
    """Delete a run."""
    if store.delete(run_id):
        print(f"Deleted run #{run_id}.")
    else:
        print(f"Run #{run_id} not found.")


def main():
    parser = argparse.ArgumentParser(description="Query stored backtest results")
    parser.add_argument("--list", action="store_true", default=True,
                        help="List recent runs (default action)")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Filter by strategy name")
    parser.add_argument("--limit", type=int, default=20,
                        help="Number of runs to show (default: 20)")
    parser.add_argument("--detail", type=int, metavar="RUN_ID",
                        help="Show full detail for a run")
    parser.add_argument("--compare", type=int, nargs="+", metavar="RUN_ID",
                        help="Side-by-side comparison of runs")
    parser.add_argument("--best", nargs=2, metavar=("METRIC", "N"),
                        help="Top N runs by metric")
    parser.add_argument("--delete", type=int, metavar="RUN_ID",
                        help="Delete a run")

    args = parser.parse_args()

    store = ResultsStore()

    if args.detail is not None:
        cmd_detail(store, args.detail)
    elif args.compare:
        cmd_compare(store, args.compare)
    elif args.best:
        cmd_best(store, args.best[0], int(args.best[1]))
    elif args.delete is not None:
        cmd_delete(store, args.delete)
    else:
        cmd_list(store, strategy=args.strategy, limit=args.limit)

    store.close()


if __name__ == "__main__":
    main()
