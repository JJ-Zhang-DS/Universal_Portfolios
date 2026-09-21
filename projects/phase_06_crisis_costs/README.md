# Phase 6 — crisis-regime cost interaction

[Project overview](../../README.md) · [Introduction](../01_introduction/README.md) · [Previous: Phase 5](../phase_05_real_data/README.md) · [Next: Phase 7](../phase_07_tax_drag/README.md)

Status: complete. Run any commands below from the repository root.

Phase 4's cost model and Phase 3's crisis regime were built independently
and never combined: Phase 4 charged a *constant* spread throughout,
including across a crisis window, even though wider spreads under stress
are well-documented (market makers demand more compensation for
adverse-selection/inventory risk). `costs.CRISIS_SPREAD_MULTIPLIER` (3x)
existed since Phase 4 but no experiment had ever exercised it. This phase
splices Phase 4's cost model onto Phase 3's calm→crisis→calm path, with
the spread 3x wider *only* during the crisis window, `n_paths=5,000`:

| Tier | Frequency | Naive (Phase 4) understates true cost by |
|---|---|---|
| mega-liquid ETF | daily | 0.04 bps/yr |
| mega-liquid ETF | monthly | 0.01 bps/yr |
| single stock | daily | 0.24 bps/yr |
| single stock | monthly | 0.05 bps/yr |

![Phase 6: how much Phase 4's constant-spread assumption understates true cost once the spread widens 3x during the crisis window](../../docs/images/phase6_crisis_cost_interaction.png)

**The understatement is real and directionally as expected (worse at
higher frequency, worse for less-liquid tiers) but small in absolute
terms** — a quarter of a bp per year at worst. The reason: the crisis
window is only 1 of the 9 total years, so even a 3x spread spike during
it gets diluted into a small blended annualized effect. A sensitivity
sweep on the multiplier itself (1x = no widening, up to 10x) confirms
this isn't fragile to that stylized assumption — even at 10x, single-
stock drag only reaches 3.5bps/yr, mega-liquid-ETF drag 0.5bps/yr, both
still trivial next to Phase 2's rebalancing premiums and Phase 3's
drawdown findings.

![Phase 6: cost-drag sensitivity to the crisis spread-widening multiplier, from 1x (no widening) to 10x](../../docs/images/phase6_crisis_multiplier_sensitivity.png)

**This closes Phase 4's flagged gap with an actual number, and the
number says the gap doesn't matter much.** Phase 3's real crisis risk —
tens of percent of extra drawdown from correlation breakdown — remains
the dominant concern; spread-widening is a real but second-order effect
on annualized growth. Worth stating plainly since it would have been easy
to assume otherwise without running the number.
