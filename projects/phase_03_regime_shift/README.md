# Phase 3 — regime-shift stress tests

[Project overview](../../README.md) · [Introduction](../01_introduction/README.md) · [Previous: Phase 2](../phase_02_mechanism_simulation/README.md) · [Next: Phase 4](../phase_04_transaction_costs/README.md)

Status: complete. Run any commands below from the repository root.

Phase 2 used *constant*-parameter GBM throughout each simulation. Real
crises don't hold parameters constant — correlation and volatility shift
mid-horizon, together, exactly when it hurts. `regime_shift_scenarios`
splices a crisis window into an otherwise-calm 9-year horizon (4y calm →
1y crisis → 4y calm) and compares against a same-length no-crisis
counterfactual (same seed, so only the crisis window's parameters differ —
common random numbers, for a low-noise comparison), for three crisis
definitions, `n_paths=5,000`:

| Crisis definition | Δ max drawdown, 50/50 CRP | Δ max drawdown, buy & hold (leg avg) | Δ growth/yr, CRP | Δ growth/yr, B&H |
|---|---|---|---|---|
| Correlation only (ρ: −0.3→0.95) | **−4.2%** | −0.1% | −0.3% | −0.0% |
| Volatility only (σ: 25%→55%) | −8.3% | −9.4% | −0.5% | **−1.4%** |
| Joint crisis (ρ,σ↑, μ turns negative) | **−28.8%** | −15.7% | −5.3% | −5.1% |

![Phase 3: drawdown and growth impact of splicing a crisis regime into an otherwise-calm horizon, vs. a same-seed no-crisis counterfactual](../../docs/images/phase3_regime_shift.png)

Two findings that directly qualify Phase 2's results:

- **Correlation-only breakdown hurts the rebalanced portfolio ~40x more
  than buy-and-hold on drawdown** (−4.2% vs. −0.1%), for essentially zero
  growth-rate cost either way. This isolates the mechanism cleanly: a
  single leg's own drawdown doesn't depend on what it's correlated with,
  but a rebalanced portfolio's risk does — directly, since the
  diversification it was relying on is what just vanished. This is
  exactly the real-world "correlations go to 1 in a crisis" phenomenon,
  and it hits rebalancing specifically, not buy-and-hold.
- **Under a realistic joint crisis, 50/50 CRP's drawdown is ~2x worse than
  buy-and-hold's** (−28.8% vs. −15.7%) despite near-identical growth-rate
  cost (−5.3% vs. −5.1%/yr). Rebalancing and buy-and-hold pay a similar
  price in terminal wealth here, but rebalancing pays it as a much sharper
  peak-to-trough hit — a distinction that matters for anything path-
  dependent (margin, redemptions, whether you hold on).

Universal Portfolio tracks fixed 50/50 CRP closely throughout this
experiment (both numbers move together in every row) — unlike Phase 2's
drift-difference test, nothing here is asymmetric enough for UP's
adaptive weighting to diverge from a static 50/50.
