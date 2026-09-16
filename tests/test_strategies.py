import numpy as np
import pytest

from universal_portfolio.cover import cover_universal_2asset
from universal_portfolio.simulate import Regime, simulate_correlated_gbm, simulate_regime_switching_gbm
from universal_portfolio.strategies import (
    bcrp_grid,
    fixed_crp_log_wealth,
    fixed_crp_log_wealth_path,
    max_drawdown_from_log_wealth_path,
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


def test_regime_switching_gbm_matches_each_segments_own_parameters():
    """Each regime segment should look, on its own, like a plain
    `simulate_correlated_gbm` draw with that segment's parameters —
    concatenation must not leak one regime's correlation/vol into
    another's."""
    n_paths = 20_000
    calm = dict(mu=(0.08, 0.08), sigma=(0.25, 0.25), rho=-0.3)
    crisis = dict(mu=(-0.25, -0.25), sigma=(0.55, 0.55), rho=0.95)

    regimes = [Regime(n_days=252, **calm), Regime(n_days=126, **crisis), Regime(n_days=252, **calm)]
    X = simulate_regime_switching_gbm(n_paths, regimes, seed=7)
    assert X.shape == (n_paths, 630, 2)

    for start, end, params in [(0, 252, calm), (252, 378, crisis), (378, 630, calm)]:
        log_r = np.log(X[:, start:end, :])
        sample_rho = np.corrcoef(log_r[:, :, 0].ravel(), log_r[:, :, 1].ravel())[0, 1]
        annualized_std = log_r.std(axis=(0, 1)) * np.sqrt(252)

        assert sample_rho == pytest.approx(params["rho"], abs=0.03)
        np.testing.assert_allclose(annualized_std, params["sigma"], atol=0.015)


def test_path_functions_final_value_matches_scalar_functions():
    """The *_path variants exist only to expose intermediate values for
    drawdown; their final column must reproduce the existing (tested)
    scalar functions exactly."""
    rng = np.random.default_rng(5)
    X = np.exp(rng.normal(loc=0.0002, scale=0.02, size=(15, 400, 2)))

    np.testing.assert_allclose(fixed_crp_log_wealth_path(X, 0.5)[:, -1], fixed_crp_log_wealth(X, 0.5))
    np.testing.assert_allclose(
        universal_portfolio_log_wealth(X, full_path=True)[:, -1], universal_portfolio_log_wealth(X, full_path=False)
    )


def test_max_drawdown_on_known_path():
    """A hand-constructed wealth path with a known peak-to-trough drop."""
    # wealth: 1 -> 1.2 -> 0.6 -> 0.9  => drawdown from peak 1.2 to trough 0.6 = -50%
    wealth = np.array([[1.0, 1.2, 0.6, 0.9]])
    log_wealth_path = np.log(wealth)

    dd = max_drawdown_from_log_wealth_path(log_wealth_path)
    assert dd[0] == pytest.approx(-0.5)


def test_max_drawdown_is_zero_for_monotonically_increasing_path():
    log_wealth_path = np.log(np.array([[1.0, 1.1, 1.3, 1.3, 1.4]]))
    dd = max_drawdown_from_log_wealth_path(log_wealth_path)
    assert dd[0] == pytest.approx(0.0)


def test_fixed_crp_daily_zero_cost_matches_frictionless_fast_path():
    """rebalance_every=1 (daily) with cost_bps=0 must exactly reproduce the
    frictionless Phase 2/3 fast-path formula: with a rebalance every single
    day, the weight is always exactly b, so the day-loop and the direct
    `b*x0 + (1-b)*x1` formula compute the same thing."""
    X = _random_paths(n_paths=15, n_days=200, seed=11)
    frictionless = fixed_crp_log_wealth_path(X, 0.5)
    zero_cost_daily = fixed_crp_log_wealth_path(X, 0.5, cost_bps=0.0, rebalance_every=1)
    np.testing.assert_allclose(zero_cost_daily[:, -1], frictionless[:, -1], rtol=1e-9)


def test_fixed_crp_zero_cost_lower_frequency_differs_from_daily():
    """Rebalancing weekly/monthly is a genuinely different portfolio
    process from daily, not just a cost saving: between rebalances the
    weight drifts away from b, so the realized daily return uses a
    different (drifted) weight than the always-exactly-b daily case. This
    difference exists even at cost_bps=0 -- confirms rebalance_every
    changes the underlying strategy, not only its cost."""
    X = _random_paths(n_paths=15, n_days=200, seed=11)
    daily = fixed_crp_log_wealth_path(X, 0.5, cost_bps=0.0, rebalance_every=1)[:, -1]
    monthly = fixed_crp_log_wealth_path(X, 0.5, cost_bps=0.0, rebalance_every=21)[:, -1]
    assert not np.allclose(daily, monthly, rtol=1e-6)


def test_fixed_crp_never_rebalancing_matches_two_leg_buy_and_hold():
    """The limiting case rebalance_every > n_days means the initial 50/50
    split is never traded again -- equivalent by definition to investing
    half the initial wealth in each leg and letting both compound
    independently, which can be computed directly without this module."""
    X = _random_paths(n_paths=15, n_days=100, seed=15)
    b = 0.5
    never_rebalanced = fixed_crp_log_wealth_path(X, b, cost_bps=0.0, rebalance_every=X.shape[1] + 1)
    final_wealth = np.exp(never_rebalanced[:, -1])

    expected = b * np.cumprod(X[:, :, 0], axis=1)[:, -1] + (1 - b) * np.cumprod(X[:, :, 1], axis=1)[:, -1]
    np.testing.assert_allclose(final_wealth, expected, rtol=1e-8)


def test_fixed_crp_cost_strictly_reduces_wealth():
    X = _random_paths(n_paths=30, n_days=250, seed=12)
    free = fixed_crp_log_wealth_path(X, 0.5, cost_bps=0.0)[:, -1]
    costed = fixed_crp_log_wealth_path(X, 0.5, cost_bps=10.0)[:, -1]
    assert np.all(costed <= free + 1e-12)
    assert np.mean(free - costed) > 0  # cost actually bites, not a no-op


def test_universal_portfolio_cost_strictly_reduces_wealth():
    X = _random_paths(n_paths=30, n_days=250, seed=13)
    free = universal_portfolio_log_wealth(X, cost_bps=0.0)
    costed = universal_portfolio_log_wealth(X, cost_bps=10.0)
    assert np.all(costed <= free + 1e-12)
    assert np.mean(free - costed) > 0


def test_less_frequent_rebalancing_reduces_cost_drag_at_fixed_cost_rate():
    """At the same per-trade cost rate, rebalancing less often means fewer
    trades and so less TOTAL cost paid -- this holds by construction
    (fewer nonzero turnover charges), independent of any market-timing
    luck. Isolates cost drag from the *structural* daily-vs-monthly
    difference confirmed in test_fixed_crp_zero_cost_lower_frequency_
    differs_from_daily by comparing each frequency's costed result against
    its OWN zero-cost baseline at that same frequency, not a shared one.
    (Whether less-frequent rebalancing is BETTER net of its own
    tracking-error cost is a separate, empirical question -- this test
    only checks the cost-drag mechanism in isolation.)"""
    X = _random_paths(n_paths=200, n_days=252, seed=14)
    cost_bps = 20.0  # exaggerated on purpose to make the drag unambiguous

    daily_free = fixed_crp_log_wealth_path(X, 0.5, cost_bps=0.0, rebalance_every=1)[:, -1]
    daily_costed = fixed_crp_log_wealth_path(X, 0.5, cost_bps=cost_bps, rebalance_every=1)[:, -1]
    monthly_free = fixed_crp_log_wealth_path(X, 0.5, cost_bps=0.0, rebalance_every=21)[:, -1]
    monthly_costed = fixed_crp_log_wealth_path(X, 0.5, cost_bps=cost_bps, rebalance_every=21)[:, -1]

    daily_drag = (daily_free - daily_costed).mean()
    monthly_drag = (monthly_free - monthly_costed).mean()
    assert monthly_drag < daily_drag
