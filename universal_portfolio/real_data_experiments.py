"""Phase 5: modern out-of-sample -- real historical price data, not
simulation.

This is a different kind of evidence from Phases 2-4: each pair below has
exactly ONE realized historical path (2016-2026), not thousands of Monte
Carlo draws. Results describe what happened over this specific decade,
not a statistically estimated expectation -- there is no standard error
to report, and a different decade could easily look different. Treat this
as an out-of-sample check on Phases 1-4's mechanism, not as independent
statistical evidence of its own.

Pairs are chosen to span the theoretical space Phase 2 mapped out:
same-sector tech pairs (NVDA/AMD, MSFT/GOOG -- expected high correlation,
negative controls) vs. cross-sector pairs anchored by WM (Waste
Management: low vol, low correlation to growth/tech -- about as boring
and diversifying a stock as exists in the US market) vs. the "classic"
equity+gold / equity+duration ETF pairs.

Phase 8 adds rolling-window robustness checks (see rolling_window_backtest
below). Phase 9 adds Phase 7's tax model, applied to these same real
pairs instead of Phase 7's synthetic symmetric-drift GBM baseline (see
tax_drag_on_real_pairs) -- testing whether a persistent-winner pair
(Phase 5/8 flagged WM/AMD and NVDA/AMD as spending most of their history
in exactly that dynamic) realizes much more tax than the calm baseline,
since trimming the winner back to target realizes a gain nearly every
rebalance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .costs import spread_cost_bps
from .market_data import price_relatives
from .strategies import (
    bcrp_grid,
    buy_and_hold_log_wealth,
    fixed_crp_log_wealth_path,
    fixed_crp_log_wealth_path_with_tax,
    max_drawdown_from_log_wealth_path,
    universal_portfolio_log_wealth,
)
from .taxes import HIGH_BRACKET, MODERATE_BRACKET, TAX_ADVANTAGED

TRADING_DAYS_PER_YEAR = 252

ETF_TICKERS = {"SPY", "QQQ", "GLD", "TLT"}

PAIRS = [
    ("SPY", "GLD"),
    ("QQQ", "TLT"),
    ("NVDA", "AMD"),
    ("MSFT", "GOOG"),
    ("WM", "TSLA"),
    ("WM", "AMD"),
    ("ISRG", "WM"),
]


def _tier_for_pair(ticker0: str, ticker1: str) -> str:
    return "mega_liquid_etf" if ticker0 in ETF_TICKERS and ticker1 in ETF_TICKERS else "single_stock"


def backtest_pair(ticker0: str, ticker1: str, prices: pd.DataFrame) -> dict:
    """Buy & hold / fixed 50/50 CRP (frictionless + Fidelity-costed) /
    BCRP (hindsight) / Universal Portfolio (frictionless + costed) on the
    real historical daily returns of one pair.

    Reuses Phases 2-4's vectorized functions with a single path
    (n_paths=1) rather than a Monte Carlo batch -- same algorithms, real
    data instead of simulated.
    """
    X = prices[[ticker0, ticker1]].to_numpy()[None, :, :]  # (1, n_days, 2)
    n_days = X.shape[1]
    tier = _tier_for_pair(ticker0, ticker1)
    cost_bps = spread_cost_bps(tier, "fidelity")

    log_r = np.log(X[0])
    realized_rho = float(np.corrcoef(log_r[:, 0], log_r[:, 1])[0, 1])
    realized_vol = np.std(log_r, axis=0) * np.sqrt(TRADING_DAYS_PER_YEAR)

    bh_growth = buy_and_hold_log_wealth(X)[0] / n_days * TRADING_DAYS_PER_YEAR
    bh_dd = [
        max_drawdown_from_log_wealth_path(np.log(X[:, :, i]).cumsum(axis=1))[0] for i in range(2)
    ]

    bcrp = bcrp_grid(X)
    crp_free_path = fixed_crp_log_wealth_path(X, 0.5)
    crp_costed_path = fixed_crp_log_wealth_path(X, 0.5, cost_bps=cost_bps, rebalance_every=1)
    up_free_path = universal_portfolio_log_wealth(X, full_path=True)
    up_costed_path = universal_portfolio_log_wealth(X, full_path=True, cost_bps=cost_bps)

    theory_excess = 0.25 * np.mean(realized_vol) ** 2 * (1 - realized_rho)

    return {
        "pair": f"{ticker0}/{ticker1}",
        "tier": tier,
        "years": round(n_days / TRADING_DAYS_PER_YEAR, 1),
        "realized_rho": realized_rho,
        "vol_0": realized_vol[0],
        "vol_1": realized_vol[1],
        "bh_0_growth": bh_growth[0],
        "bh_1_growth": bh_growth[1],
        "bh_worse_leg_max_drawdown": min(bh_dd),
        "crp_free_growth": crp_free_path[0, -1] / n_days * TRADING_DAYS_PER_YEAR,
        "crp_costed_growth": crp_costed_path[0, -1] / n_days * TRADING_DAYS_PER_YEAR,
        "crp_free_max_drawdown": max_drawdown_from_log_wealth_path(crp_free_path)[0],
        "bcrp_growth": bcrp.best_log_wealth[0] / n_days * TRADING_DAYS_PER_YEAR,
        "bcrp_weight": bcrp.best_b[0],
        "up_free_growth": up_free_path[0, -1] / n_days * TRADING_DAYS_PER_YEAR,
        "up_costed_growth": up_costed_path[0, -1] / n_days * TRADING_DAYS_PER_YEAR,
        "excess_vs_avg_leg": crp_free_path[0, -1] / n_days * TRADING_DAYS_PER_YEAR - bh_growth.mean(),
        "theory_excess_vs_avg_leg": theory_excess,
        "excess_vs_better_leg": crp_free_path[0, -1] / n_days * TRADING_DAYS_PER_YEAR - bh_growth.max(),
    }


def backtest_all_pairs(pairs=PAIRS) -> pd.DataFrame:
    prices = price_relatives()
    return pd.DataFrame([backtest_pair(t0, t1, prices) for t0, t1 in pairs])


def rolling_correlation(ticker0: str, ticker1: str, window: int = 60) -> pd.Series:
    """Rolling `window`-day realized correlation of daily log returns --
    the real-data counterpart to Phase 3's regime-switching rho.
    """
    log_r = np.log(price_relatives([ticker0, ticker1]))
    return log_r[ticker0].rolling(window).corr(log_r[ticker1])


def rolling_window_backtest(pairs=PAIRS, window_years: float = 3.0, step_days: int = 21) -> pd.DataFrame:
    """Phase 8: repeat backtest_pair over overlapping rolling windows
    instead of the single 2016-2026 full period, to check whether Phase
    5's conclusions -- especially "excess vs. the better leg is negative
    in most pairs" -- are a robust property of the mechanism or an
    artifact of this specific decade (which happened to contain some of
    the most extreme individual-stock winners in market history: NVDA,
    AMD, TSLA all multi-bagged). Reuses backtest_pair unchanged, just on
    different date slices -- same metrics, same definitions, so every
    window's numbers are directly comparable to Phase 5's single-window
    ones and to each other.

    :param window_years: length of each backtest window.
    :param step_days: how far consecutive windows' start dates are
        offset -- windows overlap heavily by design (a 3y window stepped
        21 days is >99% overlapping with its neighbor), because the goal
        is a smooth picture of how the metric evolves, not independent
        samples. Don't treat window count as an effective sample size.
    """
    prices = price_relatives()
    window_days = int(window_years * TRADING_DAYS_PER_YEAR)
    n_total = len(prices)

    rows = []
    for start in range(0, n_total - window_days + 1, step_days):
        window = prices.iloc[start : start + window_days]
        for t0, t1 in pairs:
            result = backtest_pair(t0, t1, window)
            result["window_start"] = window.index[0]
            result["window_end"] = window.index[-1]
            rows.append(result)
    return pd.DataFrame(rows)


def rolling_window_summary(df: pd.DataFrame, full_period: pd.DataFrame) -> pd.DataFrame:
    """Per-pair: how often, and by how much, would 50/50 rebalancing have
    beaten the better leg across all rolling windows -- versus the single
    full-period number Phase 5 reported, included here for direct
    comparison against the distribution it was drawn from.
    """
    full = full_period.set_index("pair")["excess_vs_better_leg"]
    rows = []
    for pair, g in df.groupby("pair", sort=False):
        rows.append(
            {
                "pair": pair,
                "n_windows": len(g),
                "full_period_excess_vs_better_leg": full.loc[pair],
                "pct_windows_beats_better_leg": (g["excess_vs_better_leg"] > 0).mean(),
                "median_excess_vs_better_leg": g["excess_vs_better_leg"].median(),
                "worst_window_excess_vs_better_leg": g["excess_vs_better_leg"].min(),
                "best_window_excess_vs_better_leg": g["excess_vs_better_leg"].max(),
            }
        )
    return pd.DataFrame(rows).set_index("pair").loc[[p for p in full.index if p in df["pair"].unique()]].reset_index()


def tax_drag_on_real_pairs(pairs=PAIRS, frequencies=None) -> pd.DataFrame:
    """Phase 9: Phase 7's tax model (fixed_crp_log_wealth_path_with_tax),
    applied to real historical pairs over their full available history
    instead of Phase 7's synthetic symmetric-drift GBM baseline.

    Phase 7 assumed mu1 == mu2 (no persistent winner). Phase 5/8 showed
    several of these real pairs spend most of their history with exactly
    that asymmetry (WM/AMD, NVDA/AMD both realized deeply negative
    excess-vs-better-leg in most rolling windows -- the fixed-weight
    portfolio was persistently trimming a winner). Trimming a winner
    realizes a gain almost every time, so tax drag on those pairs should
    run well above Phase 7's calm-regime numbers -- this checks that
    directly rather than assuming it.
    """
    if frequencies is None:
        frequencies = {"daily": 1, "weekly": 5, "monthly": 21}

    prices = price_relatives()
    scenarios = [
        (TAX_ADVANTAGED.name, 0.0),
        (f"{MODERATE_BRACKET.name}, short-term", MODERATE_BRACKET.short_term_rate),
        (f"{MODERATE_BRACKET.name}, long-term", MODERATE_BRACKET.long_term_rate),
        (f"{HIGH_BRACKET.name}, short-term", HIGH_BRACKET.short_term_rate),
        (f"{HIGH_BRACKET.name}, long-term", HIGH_BRACKET.long_term_rate),
    ]

    rows = []
    for t0, t1 in pairs:
        X = prices[[t0, t1]].to_numpy()[None, :, :]
        n_days = X.shape[1]
        for scenario_name, tax_rate in scenarios:
            for freq_name, rebalance_every in frequencies.items():
                log_wealth = fixed_crp_log_wealth_path_with_tax(
                    X, 0.5, tax_rate=tax_rate, rebalance_every=rebalance_every
                )
                rows.append(
                    {
                        "pair": f"{t0}/{t1}",
                        "scenario": scenario_name,
                        "tax_rate": tax_rate,
                        "frequency": freq_name,
                        "annualized_growth": log_wealth[0, -1] / n_days * TRADING_DAYS_PER_YEAR,
                    }
                )
    return pd.DataFrame(rows)


def tax_drag_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-pair, per-frequency: tax drag vs. the tax-advantaged (0-rate)
    baseline, at the high-bracket short-term rate -- the realistic case
    for frequent rebalancing (see strategies.fixed_crp_log_wealth_path_
    with_tax's docstring) and the scenario most exposed to a persistent
    winner's repeatedly-realized gains.
    """
    advantaged = df[df["scenario"] == TAX_ADVANTAGED.name].set_index(["pair", "frequency"])["annualized_growth"]
    taxed = df[df["scenario"] == f"{HIGH_BRACKET.name}, short-term"].set_index(["pair", "frequency"])[
        "annualized_growth"
    ]
    drag = (advantaged - taxed).rename("drag_high_bracket_short_term")
    return drag.reset_index()
