import numpy as np
import pytest

from universal_portfolio.cover import cover_universal_2asset
from universal_portfolio.simulate import simulate_correlated_gbm
from universal_portfolio.strategies import (
    bcrp_grid,
    fixed_crp_log_wealth,
    universal_portfolio_log_wealth,
)


def _random_paths(n_paths=20, n_days=300, seed=1):
    rng = np.random.default_rng(seed)
    return np.exp(rng.normal(loc=0.0002, scale=0.02, size=(n_paths, n_days, 2)))


def test_batched_universal_portfolio_matches_single_path_reference():
    """The Monte Carlo batch implementation must agree with Phase 1's
    single-path reference (`cover.cover_universal_2asset`) path-by-path —
    they're supposed to be the same algorithm, just vectorized differently.
    """
    X = _random_paths()
    batched_log_wealth = universal_portfolio_log_wealth(X)

    for i in range(X.shape[0]):
        reference = cover_universal_2asset(X[i])
        assert batched_log_wealth[i] == pytest.approx(np.log(reference.final_universal_wealth), abs=1e-8)


def test_universal_wealth_equals_mean_crp_wealth_identity_batched():
    """Same algebraic identity as Phase 1 (universal wealth == mean of all
    candidate CRPs' final wealth, true by construction), checked here in
    log-space across a whole batch of paths at once.
    """
    X = _random_paths(n_paths=50, n_days=1000, seed=2)
    grid_size = 21
    b_grid = np.linspace(0.0, 1.0, grid_size)

    log_wealth_grid = np.column_stack([fixed_crp_log_wealth(X, b) for b in b_grid])
    m = log_wealth_grid.max(axis=1, keepdims=True)
    expected = (m.squeeze(1) + np.log(np.mean(np.exp(log_wealth_grid - m), axis=1)))

    actual = universal_portfolio_log_wealth(X, grid_size=grid_size)
    np.testing.assert_allclose(actual, expected, rtol=1e-9)


def test_bcrp_is_at_least_as_good_as_any_single_grid_point():
    X = _random_paths(n_paths=30, n_days=500, seed=3)
    result = bcrp_grid(X, grid_size=21)

    for b in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert np.all(result.best_log_wealth >= fixed_crp_log_wealth(X, b) - 1e-9)


def test_universal_portfolio_never_beats_bcrp():
    """Algebraic certainty, not an empirical tendency: universal wealth is
    a (positive-weight) mean over the grid's CRP wealths, BCRP is that same
    grid's max, and mean <= max always. So Cover's regret R_T is >= 0 for
    every single path, not just on average -- this is exactly what makes
    BCRP a valid (if unreachable) benchmark."""
    X = _random_paths(n_paths=40, n_days=800, seed=4)
    bcrp = bcrp_grid(X, grid_size=21)
    up_log_wealth = universal_portfolio_log_wealth(X, grid_size=21)

    assert np.all(bcrp.best_log_wealth >= up_log_wealth - 1e-9)


def test_simulator_matches_target_moments_and_correlation():
    """Sanity check the GBM generator itself: sampled annualized drift/vol
    and shock correlation should recover the input parameters within Monte
    Carlo noise, at a large-enough sample size."""
    n_paths, n_days = 20_000, 252
    mu, sigma, rho = (0.08, -0.05), (0.25, 0.40), -0.5

    X = simulate_correlated_gbm(n_paths, n_days, mu=mu, sigma=sigma, rho=rho, seed=42)
    log_r = np.log(X)  # (n_paths, n_days, 2), daily log returns

    annualized_mean = log_r.mean(axis=(0, 1)) * 252 + 0.5 * np.array(sigma) ** 2
    annualized_std = log_r.std(axis=(0, 1)) * np.sqrt(252)
    sample_rho = np.corrcoef(log_r[:, :, 0].ravel(), log_r[:, :, 1].ravel())[0, 1]

    np.testing.assert_allclose(annualized_mean, mu, atol=0.01)
    np.testing.assert_allclose(annualized_std, sigma, atol=0.01)
    assert sample_rho == pytest.approx(rho, abs=0.02)
