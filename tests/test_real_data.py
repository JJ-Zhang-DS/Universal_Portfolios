"""Phase 5 tests. Unlike Phases 1-4 (fully synthetic/deterministic, or
pinned to a versioned pip package), these need real market data -- either
an existing local cache (data/market_prices.csv, gitignored) or a live
fetch. Skip gracefully rather than fail hard when neither is available
(e.g. a fresh clone with no network access), since that's an environment
limitation, not a code defect.
"""
import numpy as np
import pytest

from universal_portfolio.real_data_experiments import PAIRS, backtest_all_pairs, backtest_pair, rolling_correlation


@pytest.fixture(scope="module")
def prices():
    try:
        from universal_portfolio.market_data import price_relatives

        return price_relatives()
    except Exception as e:  # network unavailable, rate-limited, etc.
        pytest.skip(f"real market data unavailable: {e}")


def test_price_relatives_are_sane(prices):
    assert len(prices) > 500  # at least ~2 years of trading days
    assert (prices > 0).all().all()  # gross returns must be strictly positive
    # no single-day move should look like a data-format error (c.f. Phase 1's
    # nyse_o.csv lesson) -- bound is loose on purpose: real single-day moves
    # (e.g. AMD's observed +52%/-24%) are legitimate and must not trip this
    assert (prices < 2.0).all().all() and (prices > 0.4).all().all()


def test_backtest_pair_output_is_internally_consistent(prices):
    result = backtest_pair("SPY", "GLD", prices)

    assert -1.0 <= result["realized_rho"] <= 1.0
    assert 0.0 <= result["bcrp_weight"] <= 1.0
    assert result["vol_0"] > 0 and result["vol_1"] > 0
    # BCRP is the hindsight-optimal grid point -- it must be >= a plain 50/50 CRP
    # (0.5 is itself a candidate in the grid bcrp_grid searches over)
    assert result["bcrp_growth"] >= result["crp_free_growth"] - 1e-9
    # a real cost > 0 must not IMPROVE the costed CRP vs. its frictionless twin
    assert result["crp_costed_growth"] <= result["crp_free_growth"] + 1e-9


def test_theory_formula_direction_matches_reality(prices):
    """Not a tight numerical check (real data isn't GBM) -- just that the
    closed-form formula's SIGN and rough SCALE track the realized excess,
    the way Phase 2 established on synthetic data."""
    result = backtest_pair("SPY", "GLD", prices)
    assert result["theory_excess_vs_avg_leg"] > 0  # rho < 1 and vol > 0 here
    assert result["excess_vs_avg_leg"] > 0  # realized excess should also be positive
    ratio = result["excess_vs_avg_leg"] / result["theory_excess_vs_avg_leg"]
    assert 0.5 < ratio < 2.0  # same order of magnitude, not an exact match


def test_backtest_all_pairs_covers_every_configured_pair(prices):
    df = backtest_all_pairs()
    assert len(df) == len(PAIRS)
    assert set(df["pair"]) == {f"{t0}/{t1}" for t0, t1 in PAIRS}
    assert df["realized_rho"].between(-1.0, 1.0).all()


def test_rolling_correlation_stays_in_valid_range(prices):
    corr = rolling_correlation("QQQ", "TLT", window=60)
    valid = corr.dropna()
    assert len(valid) > 100
    assert valid.between(-1.0, 1.0).all()
