"""Correlated GBM Monte Carlo for isolating the rebalancing-premium effect
from Cover's online-learning mechanism (Phase 2 of the research plan; see
README.md).

Two assets, each geometric Brownian motion:

    d log S_i = (mu_i - sigma_i^2 / 2) dt + sigma_i dW_i,   Corr(dW_1, dW_2) = rho

Output is daily gross price relatives (dt = 1/252 by default) — the same
convention `cover.cover_universal_2asset` and `strategies.py` expect.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Regime:
    """One piecewise-constant-parameter segment of a regime-switching GBM."""

    mu: tuple[float, float]
    sigma: tuple[float, float]
    rho: float
    n_days: int


def simulate_regime_switching_gbm(
    n_paths: int,
    regimes: list[Regime],
    seed: int | None = None,
    dt: float = 1 / 252,
) -> np.ndarray:
    """Concatenate independently-drawn GBM segments, each with its own
    (mu, sigma, rho), into one path per asset.

    This is exact, not an approximation: log-returns are independent
    increments, so splicing separately-generated log-return segments with
    different parameters at the boundary *is* the definition of a
    regime-switching GBM — no continuity adjustment is needed because we
    work in return space throughout, never in price levels.

    Each regime gets its own independent random substream (via
    `np.random.SeedSequence.spawn`), not a slice of one shared stream, so
    adding/removing/reordering regimes doesn't change other regimes' draws.

    :returns: shape (n_paths, sum(r.n_days for r in regimes), 2).
    """
    seeds = np.random.SeedSequence(seed).spawn(len(regimes))
    segments = [
        simulate_correlated_gbm(n_paths, regime.n_days, mu=regime.mu, sigma=regime.sigma, rho=regime.rho, seed=s, dt=dt)
        for regime, s in zip(regimes, seeds)
    ]
    return np.concatenate(segments, axis=1)


def simulate_correlated_gbm(
    n_paths: int,
    n_days: int,
    mu: tuple[float, float],
    sigma: tuple[float, float],
    rho: float,
    seed: int | None = None,
    dt: float = 1 / 252,
) -> np.ndarray:
    """Daily gross price relatives for two correlated GBM assets.

    :param mu: (mu_1, mu_2) annualized drift.
    :param sigma: (sigma_1, sigma_2) annualized volatility.
    :param rho: correlation between the two assets' Brownian shocks.
    :returns: array of shape (n_paths, n_days, 2).
    """
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [-1, 1], got {rho}")

    rng = np.random.default_rng(seed)
    mu1, mu2 = mu
    sigma1, sigma2 = sigma

    z1 = rng.standard_normal((n_paths, n_days))
    z_indep = rng.standard_normal((n_paths, n_days))
    z2 = rho * z1 + np.sqrt(1 - rho**2) * z_indep

    log_r1 = (mu1 - 0.5 * sigma1**2) * dt + sigma1 * np.sqrt(dt) * z1
    log_r2 = (mu2 - 0.5 * sigma2**2) * dt + sigma2 * np.sqrt(dt) * z2

    X = np.empty((n_paths, n_days, 2))
    X[:, :, 0] = np.exp(log_r1)
    X[:, :, 1] = np.exp(log_r2)
    return X
