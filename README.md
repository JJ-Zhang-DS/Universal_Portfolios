# Universal Portfolios

Research replication and stress test of Thomas Cover's 1991 **Universal Portfolio** algorithm.

Universal Portfolio is a model-free allocation method: it maintains a wealth-weighted mixture of candidate constant-rebalanced portfolios (CRPs), using no look-ahead and no return-prediction model.

## Start here

The conceptual introduction is now a standalone subproject:

### [Subproject 1 — Universal Portfolio introduction](projects/01_introduction/README.md)

It explains:

- CRP, BCRP, and Universal Portfolio;
- historical-wealth weighting;
- the discrete and continuous formulas;
- why Cover's 73.62× BCRP is a hindsight benchmark while the online UP achieved 38.67×;
- and the questions tested by the simulations: volatility, correlation, drift, horizon, regime shifts, costs, taxes, and rolling-window robustness.

Use the phase guide below for detailed results; the methodology, summary, code layout, and roadmap remain here.

## Phase guide

Each completed phase has its own README, including its full results and figures.

| Phase | Topic |
|---|---|
| 1 | [replication](projects/phase_01_replication/README.md) |
| 2 | [mechanism simulation](projects/phase_02_mechanism_simulation/README.md) |
| 3 | [regime-shift stress tests](projects/phase_03_regime_shift/README.md) |
| 4 | [transaction costs](projects/phase_04_transaction_costs/README.md) |
| 5 | [modern out-of-sample](projects/phase_05_real_data/README.md) |
| 6 | [crisis-regime cost interaction](projects/phase_06_crisis_costs/README.md) |
| 7 | [tax drag](projects/phase_07_tax_drag/README.md) |
| 8 | [rolling-window robustness check](projects/phase_08_rolling_windows/README.md) |
| 9 | [tax drag on real pairs](projects/phase_09_real_pair_taxes/README.md) |

## Methodology

Four-stage arc, each stage's output feeding the next:

1. **Theory** (Phase 1) — replicate Cover's own published numbers on his own
   historical dataset, single path, no simulation, to establish a
   known-correct reference implementation before trusting it on anything else.
2. **Controlled simulation** (Phases 2-3) — vectorized Monte Carlo on
   correlated-GBM paths (`n_paths` up to 20,000), first at constant
   parameters to validate the closed-form rebalancing-premium formula
   against simulation, then with parameters that shift mid-path
   (regime-switching) to stress-test it under crisis-like conditions.
3. **Frictions** (Phases 4, 6, 7) — layer real-world costs onto the same
   simulation: sourced vendor transaction costs (Phase 4), cost widening
   under the crisis regime from stage 2 (Phase 6), and realized-gain tax
   drag under an explicit account-type/bracket/frequency model (Phase 7).
4. **Real data** (Phases 5, 8, 9) — replace simulated paths with actual
   2016-2026 daily prices for 11 tickers, re-running the *same* strategy
   functions built in stages 1-3 (not a separate implementation) on realized
   history: one full-period backtest per pair (Phase 5), then a rolling
   3-year-window version to check whether the full-period read is robust or
   an artifact of this one decade (Phase 8), then Phase 7's tax model
   applied to these same real pairs instead of synthetic data (Phase 9).

Engineering discipline held constant across all nine phases:

- **Log-wealth space throughout** — avoids overflow/underflow at 20-year
  horizons and 60%+ annualized volatility.
- **Vectorized across paths**, never a per-path Python loop — `n_paths` up
  to 20,000 in a single array operation.
- **Algebraic-identity tests before empirical ones** — e.g. `BCRP ≥
  Universal Portfolio` and `universal wealth == mean(CRP wealth)` are true
  *by construction*; a test on these catches indexing/look-ahead bugs that
  a purely numerical tolerance check would miss.
- **Common random numbers** for every before/after comparison (Phases 3, 6)
  — same seed on both sides, so the only thing that differs is the one
  parameter being tested, not Monte Carlo noise.
- **A suspiciously good result is a bug signal, not a win** — restated
  explicitly at every phase boundary; Phase 8 is the one case where this
  discipline actually caught something (a full-period result that reversed
  under robustness testing — see Phase 8).

## Results at a glance

Two findings matter most, established early and confirmed at every later stage:

1. **"Beats the average of the two legs" (what the theory guarantees) and
   "beats the better of the two legs" (what an investor actually cares
   about) are different bars — rebalancing routinely clears the first while
   failing the second.** True on synthetic data (Phase 2: positive excess
   vs. the average leg in 24/24 tested (σ,ρ) combinations, but below 50%
   win probability against the better leg in 22/24) and on real data
   (Phases 5/8: negative excess vs. the better leg in most pairs and most
   rolling windows, even for genuinely low-correlation pairs like QQQ/TLT).
2. **Tax drag dwarfs every other modeled friction by 1-2 orders of magnitude:**

   | Friction | Magnitude | Source |
   |---|---:|---|
   | Transaction costs (Fidelity, ETF / single-stock) | 0.004%–0.05%/yr | Phase 4 |
   | Crisis-widened spreads | 0.0004%–0.24%/yr | Phase 6 |
   | **Tax drag** (real pairs, high bracket, daily rebalancing) | **1.3%–13.4%/yr** | Phases 7, 9 |

Method and full numbers behind both are in the linked phase READMEs, with charts embedded inline where they were generated (`docs/images/`; regenerate anytime via the `scripts/` commands — the checked-in copies are what GitHub renders here, `results/` itself is gitignored scratch output).

## Layout

- `universal_portfolio/cover.py` — Phase 1: the reference algorithm (`cover_universal_2asset`), single-path, no external data dependency.
- `universal_portfolio/data.py` — Phase 1: NYSE(O) dataset loader + the cumulative→period conversion.
- `universal_portfolio/simulate.py` — Phase 2: correlated-GBM path generator; Phase 3: `Regime` + `simulate_regime_switching_gbm` for piecewise-constant-parameter paths.
- `universal_portfolio/strategies.py` — Phase 2: the same algorithms as `cover.py`, vectorized across simulation paths (log-wealth space throughout, for numerical stability at 20-year/60%-vol horizons); cross-checked against `cover.py` in tests. Phase 3: `*_path` variants that expose the full trajectory (not just final wealth) + `max_drawdown_from_log_wealth_path`. Phase 4: `cost_bps`/`rebalance_every` on the fixed-CRP and Universal Portfolio path functions. Phase 6: `cost_bps` accepts either a scalar or a per-day array, for a cost that varies with the regime. Phase 7: `fixed_crp_log_wealth_path_with_tax`, average-cost-basis gain/loss tracking and tax.
- `universal_portfolio/costs.py` — Phase 4: sourced commission + effective-spread assumptions by vendor and liquidity tier; the `crisis` flag and `CRISIS_SPREAD_MULTIPLIER` (defined here since Phase 4, first actually used in Phase 6).
- `universal_portfolio/taxes.py` — Phase 7: the realization-vs-wash-sale explanation, and sourced 2026 short-term/long-term/NIIT rate scenarios.
- `universal_portfolio/market_data.py` — Phase 5: real price fetch/cache (`yfinance`) + conversion to the same price-relative convention used throughout.
- `universal_portfolio/real_data_experiments.py` — Phase 5: pair backtests + rolling correlation on real data, reusing Phase 2-4's `strategies.py` functions with `n_paths=1` instead of a Monte Carlo batch. Phase 8: `rolling_window_backtest`/`rolling_window_summary`, repeating `backtest_pair` over overlapping sub-windows instead of the one full period. Phase 9: `tax_drag_on_real_pairs`/`tax_drag_summary`, applying Phase 7's `fixed_crp_log_wealth_path_with_tax` to these same real pairs.
- `universal_portfolio/experiments.py` — Phases 2-4, 6, and 7: the synthetic-data experiments, as pure functions returning DataFrames.
- `universal_portfolio/plotting.py` — chart rendering.
- `scripts/replicate_cover1991.py`, `scripts/mechanism_simulation.py`, `scripts/real_data_backtest.py` — CLI entry points.
- `tests/test_cover_replication.py` — Phase 1 tests: exact-number replication, the `universal wealth == mean(CRP wealth)` algebraic identity (true by construction, so it's what actually catches indexing/look-ahead bugs), a cross-check against `universal-portfolios`' own BCRP optimizer.
- `tests/test_strategies.py` — Phase 2-4, 6, 7 tests: batched implementation vs. Phase 1's single-path reference, the same algebraic identity batched, `BCRP ≥ Universal Portfolio` always (also algebraic, not empirical), GBM simulator calibration (including per-regime calibration), drawdown correctness on hand-constructed paths, Phase 4's cost/frequency mechanics (including a caught-by-testing subtlety: rebalancing frequency changes the underlying wealth process even at zero cost, since the weight drifts between rebalances — cost and "structural" frequency effects had to be tested separately, not conflated), Phase 6's array-valued cost (a constant array must reproduce the scalar exactly; a cost confined to a sub-window must leave the path untouched before that window starts), and Phase 7's tax mechanics (a hand-computed 2-day example verifying the realized-gain arithmetic; a caught-by-testing subtlety of its own — unlike Phase 4's cost, which is always >=0, a realized LOSS gives a tax rebate under this model's full-offset assumption, so "tax reduces wealth" only holds on AVERAGE across paths, not on every individual path, and the test had to be corrected to check the mean, not `np.all`).
- `tests/test_real_data.py` — Phase 5 tests: sane price-relative bounds (loose on purpose — AMD alone had a real +52%/-24% single day in this window), BCRP ≥ fixed CRP and costed ≤ frictionless on real data, theory-vs-reality direction/scale, rolling correlation stays in [-1, 1]. Needs a local cache or network access; skips (doesn't fail) if neither is available, since Phase 5 is inherently network-dependent in a way Phases 1-4 aren't. Phase 8 tests: rolling-window mechanics (one row per window per pair, correct window length, per-window internal consistency) and that the summary's aggregates match recomputing them directly from the raw rolling output. Phase 9 tests: real-pair tax drag is internally consistent (zero-rate baseline, every taxed scenario below it, short-term drags more than long-term at the same bracket) and the summary matches manual recomputation.
- `docs/images/` — checked-in copies of the charts embedded in the phase READMEs, one per experiment, named `phaseN_<experiment>.png`. Regenerated by the `scripts/` commands into `results/` (gitignored — treat as scratch); when a chart changes, re-copy the relevant file from `results/` into `docs/images/` so the README stays in sync.

## Planned next phases

Not yet implemented:

1. **Exact lot accounting** — replace Phase 7/9's average-cost-basis/scenario-rate simplifications with real FIFO or specific-ID lot tracking and per-lot holding periods, closing the short-term/long-term gap the current model treats as a scenario choice rather than a derived quantity. Now more clearly worth doing: Phase 9 shows the scenario choice swings the answer by 2x on some real pairs (e.g. WM/TSLA: 8.4pp short-term vs. an implied ~5pp at long-term rates), and actual lot-level holding periods on a persistent-winner pair are plausible enough to check for real, not just bound.
2. **Why does the rolling window bounce so hard around 2020-2022 for NVDA/AMD?** Phase 8 shows the swing (roughly +2pp to -29pp) but doesn't decompose it — plausibly some mix of the 2022 semiconductor selloff and shifting relative correlation/vol between the two names, not investigated here.
3. **Rolling-window tax drag** — Phase 9 used the one full 2016-2026 period per pair, the same limitation Phase 8 found in Phase 5's non-tax numbers; a rolling-window version of Phase 9 would show whether the 6x-vs-synthetic-baseline finding for NVDA/AMD is itself concentrated in specific sub-periods or holds throughout.
