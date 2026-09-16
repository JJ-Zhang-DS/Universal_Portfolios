"""Phases 5, 8, and 9: modern out-of-sample, rolling-window robustness,
and real-pair tax drag, all on real historical price data.

    python -m scripts.real_data_backtest [--out-dir results] [--refresh]

Fetches (or loads a cache of) 2016-2026 daily prices for SPY, QQQ, GLD,
TLT, NVDA, TSLA, AMD, MSFT, GOOG, ISRG, WM, backtests fixed 50/50 CRP /
BCRP / Universal Portfolio (frictionless and Fidelity-costed) on each
configured pair over the full period (Phase 5) and over rolling 3-year
sub-windows (Phase 8, checking how much the full-period numbers depend
on this specific decade), plots the rolling QQQ/TLT and WM/TSLA
correlation against the 2020 and 2022 crisis windows, and applies Phase
7's tax model to these same real pairs (Phase 9, checking whether a
persistent-winner pair realizes much more tax than Phase 7's synthetic
symmetric-drift baseline suggested).

Unlike Phases 2-4, each pair here is ONE realized historical path, not a
Monte Carlo average -- see universal_portfolio/real_data_experiments.py's
module docstring for what that does and doesn't let you conclude.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from universal_portfolio.market_data import fetch_and_cache
from universal_portfolio.plotting import (
    plot_real_pair_results,
    plot_rolling_correlation,
    plot_rolling_window_summary,
    plot_rolling_window_timeseries,
    plot_tax_drag_real_pairs,
)
from universal_portfolio.real_data_experiments import (
    backtest_all_pairs,
    rolling_correlation,
    rolling_window_backtest,
    rolling_window_summary,
    tax_drag_on_real_pairs,
    tax_drag_summary,
)

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--refresh", action="store_true", help="re-fetch prices instead of using the local cache")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.refresh:
        print("Fetching fresh price data...")
        fetch_and_cache()

    print("=" * 70)
    print("Experiment 6: real-data pair backtests (2016-2026)")
    print("=" * 70)
    df = backtest_all_pairs()
    print(df.to_string(index=False))
    df.to_csv(os.path.join(args.out_dir, "exp6_real_pairs.csv"), index=False)
    plot_real_pair_results(df, os.path.join(args.out_dir, "exp6_real_pairs.png"))

    print()
    print("Rendering rolling correlation (QQQ/TLT, WM/TSLA)...")
    qqq_tlt = rolling_correlation("QQQ", "TLT")
    wm_tsla = rolling_correlation("WM", "TSLA")
    plot_rolling_correlation(
        {"QQQ / TLT": qqq_tlt, "WM / TSLA": wm_tsla},
        os.path.join(args.out_dir, "exp6_rolling_correlation.png"),
        highlight=[
            ("2020-02-15", "2020-04-15", "COVID crash"),
            ("2022-01-01", "2022-12-31", "2022 rate-hike selloff"),
        ],
    )

    print()
    print("=" * 70)
    print("Experiment 8: rolling 3-year window robustness check")
    print("=" * 70)
    rolling = rolling_window_backtest()
    summary = rolling_window_summary(rolling, df)
    print(summary.to_string(index=False))
    rolling.to_csv(os.path.join(args.out_dir, "exp9_rolling_window.csv"), index=False)
    summary.to_csv(os.path.join(args.out_dir, "exp9_rolling_window_summary.csv"), index=False)
    plot_rolling_window_timeseries(
        rolling, df, ["NVDA/AMD", "QQQ/TLT", "ISRG/WM"],
        os.path.join(args.out_dir, "exp9_rolling_window_timeseries.png"),
    )
    plot_rolling_window_summary(summary, os.path.join(args.out_dir, "exp9_rolling_window_summary.png"))

    print()
    print("=" * 70)
    print("Experiment 9: tax drag on real pairs vs. Phase 7's synthetic baseline")
    print("=" * 70)
    tax_df = tax_drag_on_real_pairs()
    tax_summary = tax_drag_summary(tax_df)
    print(tax_summary.to_string(index=False))
    tax_df.to_csv(os.path.join(args.out_dir, "exp10_tax_drag_real_pairs.csv"), index=False)
    tax_summary.to_csv(os.path.join(args.out_dir, "exp10_tax_drag_real_pairs_summary.csv"), index=False)
    plot_tax_drag_real_pairs(tax_summary, os.path.join(args.out_dir, "exp10_tax_drag_real_pairs.png"))

    print()
    print(f"CSV and charts written to {args.out_dir}/")


if __name__ == "__main__":
    main()
