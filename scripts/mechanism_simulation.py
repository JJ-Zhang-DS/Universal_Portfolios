"""Phases 2-4: controlled Monte Carlo experiments separating the
rebalancing-premium effect from Cover's online-learning mechanism, then
stress-testing it under regime shifts and real transaction costs.

    python -m scripts.mechanism_simulation [--out-dir results] [--quick]

Runs all experiments (see universal_portfolio/experiments.py for what each
isolates), prints summary tables, saves CSVs + charts to --out-dir.
--quick cuts path counts for a fast smoke-test run (looser confidence
intervals — use the defaults for anything you'd actually cite).
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from universal_portfolio.experiments import (
    drift_difference_sweep,
    horizon_convergence,
    rebalancing_premium_sweep,
    regime_shift_scenarios,
    transaction_cost_sweep,
)
from universal_portfolio.plotting import (
    plot_drift_difference,
    plot_horizon_convergence,
    plot_rebalancing_premium,
    plot_regime_shift,
    plot_transaction_costs,
)

pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--quick", action="store_true", help="fewer Monte Carlo paths, for a fast smoke test")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    n_paths_main = 2000 if args.quick else 20_000
    n_paths_up = 500 if args.quick else 5_000

    print("=" * 70)
    print("Experiment 1: rebalancing premium vs. theory (sigma x rho sweep)")
    print("=" * 70)
    exp1 = rebalancing_premium_sweep(n_paths=n_paths_main)
    print(exp1.to_string(index=False))
    exp1.to_csv(os.path.join(args.out_dir, "exp1_rebalancing_premium.csv"), index=False)
    plot_rebalancing_premium(exp1, os.path.join(args.out_dir, "exp1_rebalancing_premium.png"))

    print()
    print("=" * 70)
    print("Experiment 2: drift-difference falsification ('permanent loser')")
    print("=" * 70)
    exp2 = drift_difference_sweep(n_paths=n_paths_up)
    print(exp2.to_string(index=False))
    exp2.to_csv(os.path.join(args.out_dir, "exp2_drift_difference.csv"), index=False)
    plot_drift_difference(exp2, os.path.join(args.out_dir, "exp2_drift_difference.png"))

    print()
    print("=" * 70)
    print("Experiment 3: horizon convergence (Cover's regret bound)")
    print("=" * 70)
    exp3 = horizon_convergence(n_paths=n_paths_up)
    print(exp3.to_string(index=False))
    exp3.to_csv(os.path.join(args.out_dir, "exp3_horizon_convergence.csv"), index=False)
    plot_horizon_convergence(exp3, os.path.join(args.out_dir, "exp3_horizon_convergence.png"))

    print()
    print("=" * 70)
    print("Experiment 4: regime-shift stress tests (correlation/vol crisis)")
    print("=" * 70)
    exp4 = regime_shift_scenarios(n_paths=n_paths_up)
    print(exp4.to_string(index=False))
    exp4.to_csv(os.path.join(args.out_dir, "exp4_regime_shift.csv"), index=False)
    plot_regime_shift(exp4, os.path.join(args.out_dir, "exp4_regime_shift.png"))

    print()
    print("=" * 70)
    print("Experiment 5: transaction costs (Fidelity vs. no price improvement)")
    print("=" * 70)
    exp5_etf = transaction_cost_sweep(tier="mega_liquid_etf", n_paths=n_paths_up)
    exp5_stock = transaction_cost_sweep(tier="single_stock", n_paths=n_paths_up)
    print("-- mega_liquid_etf tier --")
    print(exp5_etf.to_string(index=False))
    print("-- single_stock tier --")
    print(exp5_stock.to_string(index=False))
    exp5_etf.to_csv(os.path.join(args.out_dir, "exp5_costs_mega_liquid_etf.csv"), index=False)
    exp5_stock.to_csv(os.path.join(args.out_dir, "exp5_costs_single_stock.csv"), index=False)
    plot_transaction_costs(exp5_etf, exp5_stock, os.path.join(args.out_dir, "exp5_transaction_costs.png"))

    print()
    print(f"CSVs and charts written to {args.out_dir}/")


if __name__ == "__main__":
    main()
