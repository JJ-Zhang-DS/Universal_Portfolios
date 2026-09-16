"""Phase 2: controlled Monte Carlo experiments that separate the
rebalancing-premium effect from Cover's online-learning mechanism.

Three questions, each isolated by holding everything else fixed:

1. `rebalancing_premium_sweep` — does the classic diversification-return
   formula (excess_growth ~= 1/4 sigma^2 (1 - rho), symmetric case) hold
   empirically, and separately: does that translate into beating the
   better of the two assets, or only their average?
2. `drift_difference_sweep` — the direct falsification test for "just pick
   two volatile, weakly-correlated assets and rebalance": as one asset's
   drift pulls away from the other's, fixed-weight rebalancing keeps
   selling the winner to buy the loser. At what drift gap does that stop
   paying off relative to just holding the winner?
3. `horizon_convergence` — Cover's actual claim is asymptotic (regret -> 0
   as the horizon grows), not "wins in finite samples." Check the shrink
   pattern directly instead of taking that on faith.

Phase 3 adds one more question, using regime-switching (not constant-
parameter) GBM, since (1)-(3) can't represent a parameter that changes
mid-horizon:

4. `regime_shift_scenarios` — does the rebalancing premium survive when
   correlation/volatility regime-shift into a crisis exactly when a
   drawdown also hits? Three crisis definitions (correlation-only,
   volatility-only, and a realistic joint crisis where both jump together
   with drift turning negative), each compared against a no-crisis
   counterfactual over the same total horizon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .simulate import Regime, simulate_correlated_gbm, simulate_regime_switching_gbm
from .strategies import (
    bcrp_grid,
    buy_and_hold_log_wealth,
    fixed_crp_log_wealth,
    fixed_crp_log_wealth_path,
    max_drawdown_from_log_wealth_path,
    universal_portfolio_log_wealth,
)

TRADING_DAYS_PER_YEAR = 252


def rebalancing_premium_sweep(
    sigmas=(0.10, 0.20, 0.40, 0.60),
    rhos=(-0.8, -0.4, 0.0, 0.4, 0.8, 0.95),
    base_mu=0.06,
    horizon_years=10,
    n_paths=20_000,
    seed=0,
) -> pd.DataFrame:
    n_days = horizon_years * TRADING_DAYS_PER_YEAR
    rows = []
    for sigma in sigmas:
        for rho in rhos:
            X = simulate_correlated_gbm(
                n_paths, n_days, mu=(base_mu, base_mu), sigma=(sigma, sigma), rho=rho, seed=seed
            )
            bh_growth = buy_and_hold_log_wealth(X) / n_days * TRADING_DAYS_PER_YEAR  # (n_paths, 2), annualized
            crp_growth = fixed_crp_log_wealth(X, 0.5) / n_days * TRADING_DAYS_PER_YEAR  # (n_paths,)

            # theory (Fernholz excess growth rate) predicts CRP vs. the
            # WEIGHT-AVERAGED growth of the two legs, not vs. the better one
            excess_vs_avg_leg = crp_growth - bh_growth.mean(axis=1)
            # the practically relevant bar: does rebalancing beat just holding the winner?
            excess_vs_best_leg = crp_growth - bh_growth.max(axis=1)

            rows.append(
                {
                    "sigma": sigma,
                    "rho": rho,
                    "excess_vs_avg_leg_mean": excess_vs_avg_leg.mean(),
                    "excess_vs_avg_leg_se": excess_vs_avg_leg.std(ddof=1) / np.sqrt(n_paths),
                    "theory_excess": 0.25 * sigma**2 * (1 - rho),
                    "excess_vs_best_leg_mean": excess_vs_best_leg.mean(),
                    "prob_beats_best_leg": float(np.mean(crp_growth > bh_growth.max(axis=1))),
                }
            )
    return pd.DataFrame(rows)


def drift_difference_sweep(
    delta_mus=(0.0, 0.02, 0.05, 0.10, 0.20),
    sigma=0.40,
    rho=0.0,
    base_mu=0.06,
    horizon_years=10,
    n_paths=5_000,
    seed=1,
) -> pd.DataFrame:
    n_days = horizon_years * TRADING_DAYS_PER_YEAR
    rows = []
    for delta_mu in delta_mus:
        mu = (base_mu + delta_mu / 2, base_mu - delta_mu / 2)
        X = simulate_correlated_gbm(n_paths, n_days, mu=mu, sigma=(sigma, sigma), rho=rho, seed=seed)

        bh_growth = buy_and_hold_log_wealth(X) / n_days * TRADING_DAYS_PER_YEAR
        crp_growth = fixed_crp_log_wealth(X, 0.5) / n_days * TRADING_DAYS_PER_YEAR
        bcrp = bcrp_grid(X)
        up_growth = universal_portfolio_log_wealth(X) / n_days * TRADING_DAYS_PER_YEAR

        rows.append(
            {
                "delta_mu": delta_mu,
                "winner_growth": bh_growth.max(axis=1).mean(),
                "loser_growth": bh_growth.min(axis=1).mean(),
                "fixed5050_growth": crp_growth.mean(),
                "bcrp_growth": (bcrp.best_log_wealth / n_days * TRADING_DAYS_PER_YEAR).mean(),
                "universal_growth": up_growth.mean(),
                "prob_5050_beats_winner": float(np.mean(crp_growth > bh_growth.max(axis=1))),
            }
        )
    return pd.DataFrame(rows)


def horizon_convergence(
    horizons_years=(1, 5, 10, 20),
    sigma=0.40,
    rho=0.0,
    base_mu=0.06,
    n_paths=5_000,
    seed=2,
) -> pd.DataFrame:
    rows = []
    for years in horizons_years:
        n_days = years * TRADING_DAYS_PER_YEAR
        X = simulate_correlated_gbm(
            n_paths, n_days, mu=(base_mu, base_mu), sigma=(sigma, sigma), rho=rho, seed=seed
        )

        bcrp = bcrp_grid(X)
        up_log_wealth = universal_portfolio_log_wealth(X)

        # Cover's R_T = (1/T) log(S*_T / hat{S}_T); non-negative by construction
        # (universal wealth = mean of grid CRPs' wealth <= max = BCRP wealth).
        regret_per_day = (bcrp.best_log_wealth - up_log_wealth) / n_days

        rows.append(
            {
                "horizon_years": years,
                "mean_regret_per_day": regret_per_day.mean(),
                "mean_regret_per_day_se": regret_per_day.std(ddof=1) / np.sqrt(n_paths),
            }
        )
    return pd.DataFrame(rows)


def regime_shift_scenarios(
    n_paths: int = 5_000,
    pre_years: float = 4.0,
    crisis_years: float = 1.0,
    base_mu: float = 0.08,
    base_sigma: float = 0.25,
    base_rho: float = -0.3,
    seed: int = 10,
) -> pd.DataFrame:
    """Splice a crisis window into an otherwise-calm horizon and compare
    against a same-length no-crisis counterfactual, for three crisis
    definitions:

    - correlation_only: rho jumps to 0.95, mu/sigma unchanged
    - volatility_only:  sigma more than doubles, mu/rho unchanged
    - joint_crisis:     rho and sigma both jump AND drift turns sharply
      negative -- a realistic crash, where the diversification you were
      relying on (low/negative rho) disappears exactly when the drawdown
      hits, not independently of it

    The crisis and no-crisis paths for a given scenario share the same
    seed, so the pre/post (calm) segments are IDENTICAL draws in both --
    the only source of difference is the crisis window's parameters. This
    common-random-numbers pairing makes the crisis-vs-no-crisis delta far
    less noisy than comparing independent Monte Carlo runs would be.
    """
    pre_days = int(pre_years * TRADING_DAYS_PER_YEAR)
    crisis_days = int(crisis_years * TRADING_DAYS_PER_YEAR)
    total_days = 2 * pre_days + crisis_days

    calm = dict(mu=(base_mu, base_mu), sigma=(base_sigma, base_sigma), rho=base_rho)
    crisis_defs = {
        "correlation_only": dict(mu=(base_mu, base_mu), sigma=(base_sigma, base_sigma), rho=0.95),
        "volatility_only": dict(mu=(base_mu, base_mu), sigma=(0.55, 0.55), rho=base_rho),
        "joint_crisis": dict(mu=(-0.25, -0.25), sigma=(0.55, 0.55), rho=0.95),
    }

    rows = []
    for scenario_name, crisis_params in crisis_defs.items():
        for regime_label, middle in (("crisis", crisis_params), ("no_crisis", calm)):
            regimes = [
                Regime(n_days=pre_days, **calm),
                Regime(n_days=crisis_days, **middle),
                Regime(n_days=pre_days, **calm),
            ]
            X = simulate_regime_switching_gbm(n_paths, regimes, seed=seed)

            bh_log_wealth_path = np.log(X).cumsum(axis=1)  # (n_paths, n_days, 2)
            crp_log_wealth_path = fixed_crp_log_wealth_path(X, 0.5)
            up_log_wealth_path = universal_portfolio_log_wealth(X, full_path=True)

            bh_growth = bh_log_wealth_path[:, -1, :] / total_days * TRADING_DAYS_PER_YEAR  # (n_paths, 2)
            bh_dd = np.stack(
                [max_drawdown_from_log_wealth_path(bh_log_wealth_path[:, :, i]) for i in range(2)], axis=1
            )  # (n_paths, 2)

            rows.append(
                {
                    "scenario": scenario_name,
                    "regime": regime_label,
                    "crp_final_growth": (crp_log_wealth_path[:, -1] / total_days * TRADING_DAYS_PER_YEAR).mean(),
                    "crp_max_drawdown": max_drawdown_from_log_wealth_path(crp_log_wealth_path).mean(),
                    "up_final_growth": (up_log_wealth_path[:, -1] / total_days * TRADING_DAYS_PER_YEAR).mean(),
                    "up_max_drawdown": max_drawdown_from_log_wealth_path(up_log_wealth_path).mean(),
                    "bh_leg_avg_final_growth": bh_growth.mean(),
                    "bh_leg_avg_max_drawdown": bh_dd.mean(),
                }
            )
    return pd.DataFrame(rows)
