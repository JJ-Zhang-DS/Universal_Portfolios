# Universal Portfolios

Research replication of Thomas Cover's 1991 "Universal Portfolios" algorithm
(*Mathematical Finance* 1(1), 1991), triggered by a viral social-media post
that conflated the paper's hindsight-optimal benchmark with its actual
online algorithm. Full analysis: https://chatgpt.com/share/6aa97f61-57ac-83ea-8cff-fb3002fb59f2

Universal Portfolio is a **model-free, multi-asset weight-allocation**
algorithm — it does not predict returns. It maintains a wealth-weighted
mixture over a grid of candidate constant-rebalanced portfolios (CRPs) and
provably tracks the best CRP in hindsight asymptotically, without
look-ahead. This is a different problem class from single-asset return
forecasting; there is no "features" or "model training" step.

## Status: Phase 1 — replication (done)

Cover's Table 8.1 (Iroquois Brands vs. Kin Ark, NYSE, 1962-07-03 to
1984-12-31) is reproduced exactly:

| Quantity | Cover (1991) | This repo |
|---|---|---|
| Iroquois Brands buy & hold | 8.9151x | 8.91511x |
| Kin Ark buy & hold | 4.1276x | 4.12759x |
| BCRP (hindsight-optimal, 55% Iroquois) | 73.619x | 73.6190x |
| Universal Portfolio (no look-ahead) | 38.6727x | 38.6727x |

**73.6x is not achievable by any real strategy** — it's the best fixed
weight *found by looking at all 22 years of future data*. The actual
Universal Portfolio algorithm, using only information available at each
point in time, reaches 38.7x. This distinction is the whole point of the
paper, and the thing most retellings of it get wrong.

```bash
pip install -r requirements.txt
python -m scripts.replicate_cover1991
pytest
```

### Data source and a format gotcha worth knowing

Data comes from the `universal-portfolios` PyPI package's bundled
`nyse_o.csv` (36 NYSE stocks, 1962–1984), not from `yfinance` — Iroquois
Brands and Kin Ark have been delisted for decades and aren't in Yahoo
Finance's coverage.

The package ships this CSV with **no ticker-name column headers** (just
single letters) and no documented format. Naively treating each column as
a day-over-day return and multiplying it through the full series produces
nonsense (some columns "compound" to e^3000+). The columns are actually
**cumulative** price relatives indexed to day 0 (`S_t = P_t / P_0`); the
last row of a column *is* its total buy-and-hold wealth multiple directly
— don't compound it further. `universal_portfolio/data.py` handles this
conversion and documents how columns `T` (Iroquois) and `W` (Kin Ark) were
identified: by matching their last-row values against Cover's published
8.9151x / 4.1276x.

## Status: Phase 2 — mechanism simulation (done)

Controlled Monte Carlo (correlated GBM, vectorized across paths — see
`universal_portfolio/simulate.py` / `strategies.py`) isolating the
rebalancing-premium effect from Cover's online-learning mechanism:

```bash
python -m scripts.mechanism_simulation   # writes CSVs + charts to results/ (gitignored)
```

**1. Rebalancing premium vs. theory** (`sigma` × `rho` sweep, `n_paths=20,000`,
symmetric drift, 10y horizon): the closed-form diversification-return
formula `excess_growth ≈ ¼σ²(1−ρ)` (Fernholz / Booth-Fama) matches simulation
to 3+ decimal places at every one of 24 (σ, ρ) combinations — e.g. σ=60%,
ρ=0 predicts 0.0900, simulates 0.0900. **But** that excess is measured
against the *average* of the two legs. Measured against the bar that
actually matters — beating the *better* leg outright — 50/50 rebalancing
loses in most of the parameter space (`P(beats better leg)` ranges
2%–63%, and is below 50% in 22 of 24 combinations; it only clears 50%
at the highest-vol/most-negative-correlation corner). This is the
rigorous version of the "rebalancing premium ≠ beats the winner" caveat
from Phase 1.

**2. Drift-difference falsification** (σ=40% both legs, ρ=0, `n_paths=5,000`,
10y horizon): as the annualized drift gap between the two assets widens
from 0% to 20%, fixed 50/50 CRP's growth rate stays flat (≈2.1%/yr — a
direct consequence of holding `μ1+μ2` fixed while widening the gap,
confirmed by the closed-form portfolio-drift formula) while the winner
leg's growth rate climbs from 5.4% to 9.6%/yr. `P(50/50 beats winner)`
falls from 34% to 18% — this *is* the "sell the winner, keep buying the
loser" failure mode, quantified. Universal Portfolio (no look-ahead)
tracks the winner somewhat better than blind 50/50 (3.5%/yr vs. 2.1%/yr
at the widest gap) but comes nowhere near BCRP's hindsight number
(10.1%/yr) — exactly the gap Phase 3 (regret) explains.

**3. Horizon convergence**: Cover's regret `R_T = (BCRP − UP) log-wealth / T`
at σ=40%, ρ=0, `n_paths=5,000` — 0.00074 (1y) → 0.00027 (5y) → 0.00016
(10y) → 0.00010 (20y). Monotonically shrinking, consistent with Cover's
asymptotic (not finite-sample) guarantee.

## Layout

- `universal_portfolio/cover.py` — Phase 1: the reference algorithm (`cover_universal_2asset`), single-path, no external data dependency.
- `universal_portfolio/data.py` — Phase 1: NYSE(O) dataset loader + the cumulative→period conversion.
- `universal_portfolio/simulate.py` — Phase 2: correlated-GBM path generator.
- `universal_portfolio/strategies.py` — Phase 2: the same algorithms as `cover.py`, vectorized across simulation paths (log-wealth space throughout, for numerical stability at 20-year/60%-vol horizons); cross-checked against `cover.py` in tests.
- `universal_portfolio/experiments.py` — Phase 2: the three experiments above, as pure functions returning DataFrames.
- `universal_portfolio/plotting.py` — Phase 2: chart rendering.
- `scripts/replicate_cover1991.py`, `scripts/mechanism_simulation.py` — CLI entry points.
- `tests/test_cover_replication.py` — Phase 1 tests: exact-number replication, the `universal wealth == mean(CRP wealth)` algebraic identity (true by construction, so it's what actually catches indexing/look-ahead bugs), a cross-check against `universal-portfolios`' own BCRP optimizer.
- `tests/test_strategies.py` — Phase 2 tests: batched implementation vs. Phase 1's single-path reference, the same algebraic identity batched, `BCRP ≥ Universal Portfolio` always (also algebraic, not empirical), and GBM simulator calibration.

## Planned next phases

Not yet implemented:

1. **Regime-change stress tests** — correlation suddenly → 1, volatility regime shift mid-horizon. Phase 2's drift-difference sweep already falsifies "just rebalance any two volatile stocks" for *persistent* winner/loser; this extends it to parameters that *change*, which constant-parameter GBM can't represent.
2. **Transaction costs** — proportional cost + rebalancing frequency (daily/weekly/monthly), since both Phase 1 and Phase 2 are frictionless and daily rebalancing is not free.
3. **Modern out-of-sample** — real ETF pairs. Important caveat given Phase 2's results: candidates must be genuinely weakly/negatively correlated (e.g. equity+duration or equity+gold), not two single-name growth stocks, and *especially* not two tickers both carrying US-equity-market beta — correlation between those tends toward 1 exactly when it matters most (drawdowns), which is (1) above.
