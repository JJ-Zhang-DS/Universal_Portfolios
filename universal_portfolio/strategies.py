"""Vectorized (across simulation paths) strategy evaluators.

Everything works in log-wealth space and only exponentiates at the end.
Over a 20-year, 60%-vol horizon a handful of candidate CRPs can swing
final wealth by many orders of magnitude, so summing log-returns (and,
for the wealth-weighted mixture in `universal_portfolio_batch`, using the
standard log-sum-exp shift) avoids overflow that a naive running product
would hit.

`cover.cover_universal_2asset` (Phase 1) is a single-path, unvectorized
reference implementation kept intentionally simple/obviously-correct for
the historical replication. This module re-implements the same algorithm
vectorized across a paths dimension for Monte Carlo use; the two are
cross-checked for agreement in tests/test_strategies.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def buy_and_hold_log_wealth(X: np.ndarray) -> np.ndarray:
    """Final log-wealth of holding each asset alone, no rebalancing.

    :param X: shape (n_paths, n_days, 2).
    :returns: shape (n_paths, 2).
    """
    return np.log(X).sum(axis=1)


def fixed_crp_log_wealth(X: np.ndarray, b: float) -> np.ndarray:
    """Final log-wealth of a constant-rebalanced portfolio with fixed
    weight `b` on asset 0 (rebalanced every day).

    :param X: shape (n_paths, n_days, 2).
    :returns: shape (n_paths,).
    """
    growth = b * X[:, :, 0] + (1 - b) * X[:, :, 1]
    return np.log(growth).sum(axis=1)


@dataclass
class BCRPResult:
    best_b: np.ndarray  # shape (n_paths,)
    best_log_wealth: np.ndarray  # shape (n_paths,)


def bcrp_grid(X: np.ndarray, grid_size: int = 21) -> BCRPResult:
    """Best Constant Rebalanced Portfolio per path (hindsight optimum),
    grid-searched over `grid_size` evenly spaced weights on [0, 1].

    Loops over the (small, default 21-point) grid rather than materializing
    a (n_paths, n_days, grid_size) array, to keep memory bounded for large
    Monte Carlo batches.
    """
    b_grid = np.linspace(0.0, 1.0, grid_size)
    n_paths = X.shape[0]
    log_wealth = np.empty((n_paths, grid_size))

    for j, b in enumerate(b_grid):
        log_wealth[:, j] = fixed_crp_log_wealth(X, b)

    best_idx = np.argmax(log_wealth, axis=1)
    rows = np.arange(n_paths)
    return BCRPResult(best_b=b_grid[best_idx], best_log_wealth=log_wealth[rows, best_idx])


def universal_portfolio_log_wealth(X: np.ndarray, grid_size: int = 21) -> np.ndarray:
    """Cover's Universal Portfolio, vectorized across paths.

    Same algorithm as `cover.cover_universal_2asset` (wealth-weighted
    mixture over a grid of candidate CRPs, using only wealth accumulated
    through yesterday) but batched: the only unvectorized loop is over
    days (n_days iterations), not over paths.

    :param X: shape (n_paths, n_days, 2).
    :returns: final log-wealth, shape (n_paths,).
    """
    n_paths, n_days, _ = X.shape
    b_grid = np.linspace(0.0, 1.0, grid_size)

    log_crp_wealth = np.zeros((n_paths, grid_size))
    log_universal_wealth = np.zeros(n_paths)

    for t in range(n_days):
        # log-sum-exp shift: keeps exp() arguments <= 0 regardless of how
        # large log_crp_wealth has grown, so this never overflows.
        m = log_crp_wealth.max(axis=1, keepdims=True)
        w = np.exp(log_crp_wealth - m)
        prob = w / w.sum(axis=1, keepdims=True)
        b_hat = (prob * b_grid).sum(axis=1)

        x0, x1 = X[:, t, 0], X[:, t, 1]
        log_universal_wealth += np.log(b_hat * x0 + (1 - b_hat) * x1)
        log_crp_wealth += np.log(b_grid * x0[:, None] + (1 - b_grid) * x1[:, None])

    return log_universal_wealth
