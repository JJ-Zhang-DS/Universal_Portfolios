"""Cover (1991) Universal Portfolio for the two-asset case.

Implements the exact algorithm from Section 8 of T. Cover, "Universal
Portfolios", Mathematical Finance 1(1), 1991
(https://isl.stanford.edu/~cover/papers/paper93.pdf): a discrete grid of
b in {0, 1/20, ..., 1} (21 constant-rebalanced portfolios by default), with
each day's universal weight set to the wealth-weighted average of the grid,
using only wealth accumulated through yesterday.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CoverResult:
    b_grid: np.ndarray  # candidate CRP weights on asset 0, shape (grid_size,)
    b_hat: np.ndarray  # universal portfolio's daily weight on asset 0, shape (T,)
    universal_wealth: np.ndarray  # universal portfolio's cumulative wealth, shape (T,)
    crp_wealth: np.ndarray  # each candidate CRP's FINAL wealth, shape (grid_size,)

    @property
    def best_crp_weight(self) -> float:
        return float(self.b_grid[np.argmax(self.crp_wealth)])

    @property
    def bcrp_wealth(self) -> float:
        """Best Constant Rebalanced Portfolio — hindsight optimum, not a tradeable strategy."""
        return float(self.crp_wealth.max())

    @property
    def final_universal_wealth(self) -> float:
        return float(self.universal_wealth[-1])


def cover_universal_2asset(X: np.ndarray, grid_size: int = 21) -> CoverResult:
    """Run Cover's discretized Universal Portfolio on a 2-asset return series.

    :param X: shape (T, 2) array of daily gross price relatives
        (e.g. 1.02 == +2% that day). Column 0 is the asset `b_grid`/`b_hat`
        are weights *on*; column 1 gets the remainder (1 - b).
    :param grid_size: number of evenly spaced candidate CRPs on [0, 1].
        Cover's own Section 8 example uses 21 (step 0.05), the default here.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"X must have shape (T, 2), got {X.shape}")

    b_grid = np.linspace(0.0, 1.0, grid_size)
    crp_wealth = np.ones(grid_size)
    universal_wealth = np.empty(len(X))
    b_hat = np.empty(len(X))
    wealth = 1.0

    for t in range(len(X)):
        x0, x1 = X[t]

        # today's weight uses only wealth accumulated through t-1 — no look-ahead
        strategy_prob = crp_wealth / crp_wealth.sum()
        b_hat[t] = np.sum(strategy_prob * b_grid)

        wealth *= b_hat[t] * x0 + (1 - b_hat[t]) * x1
        universal_wealth[t] = wealth

        crp_wealth *= b_grid * x0 + (1 - b_grid) * x1

    return CoverResult(b_grid=b_grid, b_hat=b_hat, universal_wealth=universal_wealth, crp_wealth=crp_wealth)
