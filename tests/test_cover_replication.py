import numpy as np
import pandas as pd
import pytest

from universal_portfolio.cover import cover_universal_2asset
from universal_portfolio.data import load_iroquois_kin_ark


def test_universal_wealth_equals_mean_crp_wealth_identity():
    """Uniform-prior telescoping identity: universal wealth == mean of all
    candidate CRPs' final wealth. This holds by construction for ANY input
    data (it's an algebraic property of the wealth-weighted mixture, not an
    empirical result), so it's what actually catches indexing / look-ahead
    bugs, independent of whichever dataset you point the algorithm at.
    """
    rng = np.random.default_rng(0)
    X = np.exp(rng.normal(loc=0.0, scale=0.02, size=(500, 2)))

    result = cover_universal_2asset(X)

    assert result.final_universal_wealth == pytest.approx(result.crp_wealth.mean(), rel=1e-9)


def test_replicates_cover_1991_table_8_1():
    """Exact replication of Cover (1991) Section 8 / Table 8.1: Iroquois
    Brands vs. Kin Ark, NYSE(O) dataset, 1962-07-03 to 1984-12-31 (5651
    trading days). See data.py's module docstring for how these two
    columns were identified in the bundled, otherwise-unlabeled CSV.
    """
    X = load_iroquois_kin_ark().to_numpy()

    iroquois_wealth, kin_ark_wealth = X.prod(axis=0)
    assert iroquois_wealth == pytest.approx(8.9151, rel=1e-4)
    assert kin_ark_wealth == pytest.approx(4.1276, rel=1e-4)

    result = cover_universal_2asset(X)

    # Best Constant Rebalanced Portfolio: hindsight optimum, not achievable
    # without seeing the future. This is Cover's benchmark, not a strategy.
    assert result.best_crp_weight == pytest.approx(0.55)
    assert result.bcrp_wealth == pytest.approx(73.619, rel=1e-4)

    # Cover's actual Universal Portfolio: no look-ahead.
    assert result.final_universal_wealth == pytest.approx(38.6727, rel=1e-4)


def test_bcrp_weight_matches_independent_convex_optimizer():
    """Cross-check the 21-point grid-search BCRP weight against the
    `universal-portfolios` package's own continuous (cvxopt-based) BCRP
    optimizer, as a second, independently-implemented correctness check
    beyond the algebraic identity above. Expected to differ only by
    (much less than) the grid's 0.05 resolution.
    """
    from universal import tools

    X_df = load_iroquois_kin_ark()
    result = cover_universal_2asset(X_df.to_numpy())

    continuous_best_b = tools.opt_weights(X_df, metric="return", max_leverage=1)[0]

    assert result.best_crp_weight == pytest.approx(continuous_best_b, abs=0.02)
