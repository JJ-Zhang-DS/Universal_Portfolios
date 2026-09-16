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


def fixed_crp_log_wealth_path(
    X: np.ndarray, b: float, cost_bps: float | np.ndarray = 0.0, rebalance_every: int = 1
) -> np.ndarray:
    """Same as `fixed_crp_log_wealth` but returns the full cumulative
    log-wealth trajectory instead of just the final value — needed for
    drawdown, which depends on the path, not just the endpoint. Also
    adds Phase 4's transaction costs and rebalancing frequency.

    :param cost_bps: one-way turnover cost in basis points (see costs.py),
        charged at each rebalance on |target_weight b - drifted_weight|.
        Either a scalar (constant throughout) or an array of length
        n_days (Phase 6: e.g. wider during a crisis window, narrower
        outside it — see costs.spread_cost_bps's `crisis` flag). 0
        (default) reproduces the frictionless Phase 2/3 behavior exactly.
    :param rebalance_every: rebalance to `b` every N days; the weight
        drifts with realized returns in between. 1 (default) = daily,
        matching Phase 2/3.
    :returns: shape (n_paths, n_days).
    """
    n_days = X.shape[1]
    cost_bps_arr = np.broadcast_to(np.asarray(cost_bps, dtype=float), (n_days,))

    if np.all(cost_bps_arr == 0.0) and rebalance_every == 1:
        growth = b * X[:, :, 0] + (1 - b) * X[:, :, 1]
        return np.log(growth).cumsum(axis=1)

    n_paths = X.shape[0]
    w = np.full(n_paths, b)
    log_wealth_path = np.empty((n_paths, n_days))
    running = np.zeros(n_paths)

    for t in range(n_days):
        x0, x1 = X[:, t, 0], X[:, t, 1]
        gross = w * x0 + (1 - w) * x1
        running = running + np.log(gross)
        w = (w * x0) / gross  # drift with realized returns, no trading yet

        if t % rebalance_every == rebalance_every - 1:
            turnover = np.abs(b - w)
            cost_frac = cost_bps_arr[t] / 10_000.0
            running = running + np.log(np.maximum(1 - cost_frac * turnover, 1e-6))
            w = np.full(n_paths, b)

        log_wealth_path[:, t] = running

    return log_wealth_path


def max_drawdown_from_log_wealth_path(log_wealth_path: np.ndarray, axis: int = -1) -> np.ndarray:
    """Max drawdown (as a negative fraction, e.g. -0.35 == -35%) from a
    cumulative log-wealth trajectory.

    Works entirely on differences from the running max in log-space
    (`log(W_t / running_max(W)_t)`) and only exponentiates that bounded,
    typically-small difference — not the raw cumulative log-wealth, which
    can otherwise be large enough that `exp` overflows. Since `exp` is
    monotonic, `running_max(exp(L)) == exp(running_max(L))`, so this is
    exact, not an approximation.
    """
    running_max = np.maximum.accumulate(log_wealth_path, axis=axis)
    log_drawdown = log_wealth_path - running_max  # <= 0 everywhere
    return np.exp(log_drawdown.min(axis=axis)) - 1


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


def universal_portfolio_log_wealth(
    X: np.ndarray, grid_size: int = 21, full_path: bool = False, cost_bps: float | np.ndarray = 0.0
) -> np.ndarray:
    """Cover's Universal Portfolio, vectorized across paths.

    Same algorithm as `cover.cover_universal_2asset` (wealth-weighted
    mixture over a grid of candidate CRPs, using only wealth accumulated
    through yesterday) but batched: the only unvectorized loop is over
    days (n_days iterations), not over paths.

    :param X: shape (n_paths, n_days, 2).
    :param full_path: if True, return the full cumulative log-wealth
        trajectory (shape (n_paths, n_days)) instead of just the final
        value — needed for drawdown, which depends on the path.
    :param cost_bps: Phase 4: one-way turnover cost in basis points,
        charged daily on |b_hat_t - drifted_weight| (UP's target weight
        changes every day by construction, so "rebalancing frequency"
        isn't a separate lever here the way it is for a fixed CRP — see
        README). Either a scalar or an array of length n_days (Phase 6:
        time-varying, e.g. wider during a crisis window). 0 (default)
        reproduces the frictionless Phase 1-3 behavior exactly. No cost
        on day 0 (initial allocation from cash, not a trade against an
        existing position).
    :returns: shape (n_paths,), or (n_paths, n_days) if `full_path`.
    """
    n_paths, n_days, _ = X.shape
    b_grid = np.linspace(0.0, 1.0, grid_size)
    cost_frac_arr = np.broadcast_to(np.asarray(cost_bps, dtype=float), (n_days,)) / 10_000.0

    log_crp_wealth = np.zeros((n_paths, grid_size))
    log_universal_wealth_path = np.empty((n_paths, n_days))
    running = np.zeros(n_paths)
    w_drifted = None

    for t in range(n_days):
        # log-sum-exp shift: keeps exp() arguments <= 0 regardless of how
        # large log_crp_wealth has grown, so this never overflows.
        m = log_crp_wealth.max(axis=1, keepdims=True)
        w = np.exp(log_crp_wealth - m)
        prob = w / w.sum(axis=1, keepdims=True)
        b_hat = (prob * b_grid).sum(axis=1)

        if cost_frac_arr[t] and w_drifted is not None:
            turnover = np.abs(b_hat - w_drifted)
            running = running + np.log(np.maximum(1 - cost_frac_arr[t] * turnover, 1e-6))

        x0, x1 = X[:, t, 0], X[:, t, 1]
        gross = b_hat * x0 + (1 - b_hat) * x1
        running = running + np.log(gross)
        log_universal_wealth_path[:, t] = running
        w_drifted = (b_hat * x0) / gross

        log_crp_wealth += np.log(b_grid * x0[:, None] + (1 - b_grid) * x1[:, None])

    return log_universal_wealth_path if full_path else log_universal_wealth_path[:, -1]


def fixed_crp_log_wealth_path_with_tax(
    X: np.ndarray, b: float, tax_rate: float = 0.0, rebalance_every: int = 1
) -> np.ndarray:
    """Fixed-weight CRP with US capital-gains tax on every rebalancing
    sale (Phase 7), average-cost-basis method.

    On a rebalance day, whichever leg is overweight gets sold down to
    target; realized gain/loss = amount_sold - (fraction_of_position_sold
    * that leg's cost basis). Only one leg is ever sold on a given
    rebalance (the proceeds fund the other leg's purchase), so exactly
    one of the two legs' gains is nonzero per rebalance, by construction.
    The resulting tax is funded by scaling BOTH legs' value AND basis
    down proportionally -- modeling it as a small pro-rata sale across
    the whole portfolio, whose OWN embedded gain/loss this function does
    not recursively re-tax (a deliberate second-order simplification: the
    point is to capture realization-driven tax drag, not reproduce exact
    IRS lot accounting). A realized LOSS produces a negative tax (a
    rebate) -- optimistic versus the real $3,000/yr ordinary-income
    offset cap with carryforward, since it assumes full offset against
    other income/gains that year.

    `tax_rate` is a single combined rate per call, not derived from
    actually tracking each lot's holding period (intractable for a
    portfolio that trades on every rebalance) -- callers choose a
    short-term (ordinary-income) or long-term (preferential) rate as a
    scenario, per taxes.py's TaxScenario. 0 reproduces the frictionless
    Phase 2/3 behavior exactly (this is also what a tax-advantaged
    account, e.g. an IRA, looks like -- trades inside it aren't taxable
    events at all).

    :param X: shape (n_paths, n_days, 2).
    :returns: log-wealth, shape (n_paths, n_days).
    """
    n_paths, n_days, _ = X.shape

    value0 = np.full(n_paths, b)
    value1 = np.full(n_paths, 1.0 - b)
    basis0 = np.full(n_paths, b)
    basis1 = np.full(n_paths, 1.0 - b)
    log_wealth_path = np.empty((n_paths, n_days))

    for t in range(n_days):
        x0, x1 = X[:, t, 0], X[:, t, 1]
        value0 = value0 * x0
        value1 = value1 * x1  # basis is unaffected by price movement, only by transactions

        if t % rebalance_every == rebalance_every - 1:
            total = value0 + value1
            target0 = b * total

            sell0 = np.maximum(value0 - target0, 0.0)
            buy0 = np.maximum(target0 - value0, 0.0)
            frac_sold0 = np.where(value0 > 0, sell0 / np.where(value0 > 0, value0, 1.0), 0.0)
            realized_basis0 = frac_sold0 * basis0
            gain0 = sell0 - realized_basis0

            sell1 = buy0  # leg 1 is the mirror image: whichever leg isn't sold funds the other's purchase
            buy1 = sell0
            frac_sold1 = np.where(value1 > 0, sell1 / np.where(value1 > 0, value1, 1.0), 0.0)
            realized_basis1 = frac_sold1 * basis1
            gain1 = sell1 - realized_basis1

            tax_owed = tax_rate * (gain0 + gain1)

            basis0 = basis0 - realized_basis0 + buy0
            basis1 = basis1 - realized_basis1 + buy1
            value0, value1 = target0, total - target0

            scale = np.maximum((total - tax_owed) / total, 1e-6)
            value0, value1 = value0 * scale, value1 * scale
            basis0, basis1 = basis0 * scale, basis1 * scale

        log_wealth_path[:, t] = np.log(value0 + value1)

    return log_wealth_path
