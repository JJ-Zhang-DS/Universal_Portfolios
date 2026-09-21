# Phase 8 — rolling-window robustness check

[Project overview](../../README.md) · [Introduction](../01_introduction/README.md) · [Previous: Phase 7](../phase_07_tax_drag/README.md) · [Next: Phase 9](../phase_09_real_pair_taxes/README.md)

Status: complete. Run any commands below from the repository root.

Phase 5 ran ONE full-period (2016-2026) backtest per pair — a single
point estimate, from a decade that happened to contain some of the most
extreme individual-stock winners in market history (NVDA, AMD, TSLA all
multi-bagged). `rolling_window_backtest` reuses Phase 5's `backtest_pair`
unchanged, just on overlapping 3-year sub-windows (stepped monthly, 93
windows per pair — heavily overlapping by design, not 93 independent
samples; see the function's own docstring) to check whether Phase 5's
headline finding — "excess vs. the better leg is negative in most
pairs" — is a robust property of the mechanism or an artifact of this
one window.

| Pair | Full-period excess vs. better leg | % of rolling windows that beat it | Median rolling excess |
|---|---|---|---|
| ISRG/WM | +0.2pp (positive) | 28% | −1.6pp |
| WM/TSLA | −2.9pp | 24% | −6.5pp |
| WM/AMD | −12.5pp | 17% | −9.1pp |
| SPY/GLD | −0.0pp (flat) | 10% | −2.0pp |
| **NVDA/AMD** | **+1.2pp (positive)** | **3%** | **−10.0pp** |
| **MSFT/GOOG** | **+0.2pp (positive)** | **3%** | **−2.9pp** |
| QQQ/TLT | −8.6pp | 1% | −8.8pp |

![Phase 8: full-period excess vs. the better leg against the distribution of excess across 93 rolling 3-year windows, per pair](../../docs/images/phase8_rolling_window_summary.png)

**The finding is not just confirmed, it's sharpened, and one part of
Phase 5's read needs correcting.** Every single pair's MEDIAN rolling-
window result is negative, and no pair beats the better leg in a
majority of realistic 3-year holding periods — even the least-bad pair
(ISRG/WM) only clears the bar 28% of the time. But look at the two bolded
rows: NVDA/AMD and MSFT/GOOG were the two pairs Phase 5 flagged as
genuine wins (positive full-period excess). Rolling windows show they
beat the better leg in only **3% of 3-year periods each**, with medians
of −10.0pp and −2.9pp. The full-period "win" for both was a real number,
correctly computed, but not remotely representative of what most 3-year
holders of that pair over this decade actually experienced — it was
being driven by the specific start/end points of the one window tested.

The NVDA/AMD time series makes this unambiguous: rolling excess ranges
from roughly +2pp down to **−29pp**, spending most of 2018-2023 firmly
negative (often below −15pp), and only turns positive in the handful of
3-year windows ending in the most recent AI-driven rally. The full-period
number landed in exactly that narrow favorable tail. QQQ/TLT, by
contrast, is consistently negative across nearly the entire decade —
that finding was never fragile to begin with, and this confirms it.

![Phase 8: rolling 3-year excess vs. the better leg over time, for NVDA/AMD, QQQ/TLT, and ISRG/WM](../../docs/images/phase8_rolling_window_timeseries.png)

**Practical upshot**: don't trust a single full-period backtest for a
pair-rebalancing decision, even a real, honestly-computed one — check
the rolling distribution first. A number that looks good over one
specific decade can be almost entirely a function of where that decade
happened to start and end.
