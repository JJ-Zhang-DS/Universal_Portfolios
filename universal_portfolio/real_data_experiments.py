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
    max_drawdown_from_log_wealth_path,
    universal_portfolio_log_wealth,
)

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
