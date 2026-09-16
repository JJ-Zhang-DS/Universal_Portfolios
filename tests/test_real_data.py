"""Phase 5 tests. Unlike Phases 1-4 (fully synthetic/deterministic, or
pinned to a versioned pip package), these need real market data -- either
an existing local cache (data/market_prices.csv, gitignored) or a live
fetch. Skip gracefully rather than fail hard when neither is available
(e.g. a fresh clone with no network access), since that's an environment
limitation, not a code defect.
"""
import numpy as np
import pytest

from universal_portfolio.real_data_experiments import (
    PAIRS,
    backtest_all_pairs,
    backtest_pair,
    rolling_correlation,
    rolling_window_backtest,
    rolling_window_summary,
)


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


def test_rolling_window_backtest_produces_one_row_per_pair_per_window(prices):
    """Phase 8: coarse step/short window, just to check the windowing
    mechanics -- one row per (window, pair), correct window length, and
    the same internal-consistency properties backtest_pair already
    guarantees hold for every individual window too."""
    two_pairs = PAIRS[:2]
    df = rolling_window_backtest(pairs=two_pairs, window_years=1.0, step_days=126)

    assert len(df) > 0
    assert len(df) % len(two_pairs) == 0  # every window covers every requested pair
    assert set(df["pair"]) == {f"{t0}/{t1}" for t0, t1 in two_pairs}
    assert (df["years"] - 1.0).abs().max() < 0.05  # each window is ~1 year, per window_years
    assert df["realized_rho"].between(-1.0, 1.0).all()
    # costed can't beat frictionless in any individual window either
    assert (df["crp_costed_growth"] <= df["crp_free_growth"] + 1e-9).all()


def test_rolling_window_summary_matches_manual_calculation(prices):
    """The summary's percentages/medians must match recomputing them
    directly from the raw rolling DataFrame -- not just plausible-looking
    output from rolling_window_summary's own (separate) aggregation path."""
    one_pair = PAIRS[:1]
    rolling = rolling_window_backtest(pairs=one_pair, window_years=1.0, step_days=126)
    full = backtest_all_pairs(pairs=one_pair)

    summary = rolling_window_summary(rolling, full)
    assert len(summary) == 1

    row = summary.iloc[0]
    pair_label = f"{one_pair[0][0]}/{one_pair[0][1]}"
    assert row["pair"] == pair_label
    assert row["n_windows"] == len(rolling)
    expected_pct = (rolling["excess_vs_better_leg"] > 0).mean()
    assert row["pct_windows_beats_better_leg"] == pytest.approx(expected_pct)
    assert row["median_excess_vs_better_leg"] == pytest.approx(rolling["excess_vs_better_leg"].median())
    assert row["full_period_excess_vs_better_leg"] == pytest.approx(
        full.set_index("pair").loc[pair_label, "excess_vs_better_leg"]
    )
