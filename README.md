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

## Status: Phase 3 — regime-shift stress tests (done)

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

## Status: Phase 4 — transaction costs (done)

Phases 1-3 are all frictionless. Real rebalancing pays a cost, and — this
is the correction worth leading with — **for Fidelity, on US stock/ETF
trades, that cost is not commission.** Commission has been $0 industry-wide
(Fidelity, Schwab, Vanguard, Robinhood, E\*TRADE) since 2019; Fidelity's
sole 2026 exception (up to $100 on ~120 niche ETFs whose issuers don't
fund its revenue-sharing program, effective June 2026) explicitly excludes
SPY (SPDR), QQQ (Invesco), GLD (SPDR), and TLT (iShares) — all named as
unaffected. The real cost is exactly what you flagged: **the bid-ask
spread** — you pay the ask buying, receive the bid selling, and that gap
doesn't show up as a line-item commission. `universal_portfolio/costs.py`
sources this from Fidelity's own Q1 2026 execution-quality disclosure
(SEC Rule 605 data, Apr 2025–Mar 2026, [fidelity.com/trading/execution-quality/overview](https://www.fidelity.com/trading/execution-quality/overview)):
95.37% of shares get price improvement over the NBBO, and the average
**effective** spread — the actual gap between the NBBO midpoint and your
execution price, already net of that price improvement, i.e. what you
really pay, not the quoted spread — is **$0.0056/share**, across
Fidelity's whole Rule-605-eligible order flow (not broken out per ticker,
so not literally a SPY/QQQ-specific number). Commission facts from
[fidelity.com/trading/commissions-margin-rates](https://www.fidelity.com/trading/commissions-margin-rates), checked Sep 2026.

From that anchor plus the well-documented fact that SPY/QQQ trade with
penny-wide NBBO quoted spreads (~0.15-0.2bps at current prices), this
repo assigns its own (conservative, rounded-up, clearly-labeled-as-
assumption) one-way cost tiers: mega-liquid ETFs 0.3bps at Fidelity /
0.7bps with no price improvement (a stand-in for a lower-execution-
quality vendor, not a specific verified competitor), single stocks
2.0bps / 4.0bps. `n_paths=5,000`, calm-regime (σ=25%, ρ=−0.3) parameters
matching Phase 3, 10y horizon:

| Tier | Frequency | Fidelity drag | No-price-improvement drag |
|---|---|---|---|
| mega-liquid ETF (SPY/QQQ-like) | daily | 0.4 bps/yr | 0.9 bps/yr |
| mega-liquid ETF | monthly | 0.0 bps/yr | 0.2 bps/yr |
| single stock | daily | 2.6 bps/yr | 5.1 bps/yr |
| single stock | monthly | 0.5 bps/yr | 1.1 bps/yr |

(drag = frictionless-daily annualized growth minus the costed scenario's)

**The answer to the original concern**: for SPY/QQQ-tier liquidity at
Fidelity, the bid-ask spread is real but small enough to be a non-issue
for daily rebalancing — under 1bp/year, trivial next to Phase 2's
rebalancing premiums (which ran tens to hundreds of bps/year at realistic
vol). For single-name stocks it's more material (up to ~5bps/yr daily)
but still modest. In every scenario tested, **monthly rebalancing
recovers most of the daily cost drag** — confirming the original
analysis's suspicion that more-frequent isn't automatically better once
costs are counted, though here it's a small effect because the costs
themselves are small at this tier.

**What this deliberately doesn't cover**: spreads widen during the
crisis regime Phase 3 modeled (a well-documented market-microstructure
effect — market makers demand more compensation for adverse-selection
risk under stress) — this repo doesn't yet quantify that interaction.
More importantly, for a **taxable account**, daily rebalancing realizes
short-term capital gains every trading day, taxed as ordinary income —
almost certainly a far larger cost than the bid-ask spread for a real
investor, and entirely out of scope here (this phase covers execution
cost, not tax drag).

## Layout

- `universal_portfolio/cover.py` — Phase 1: the reference algorithm (`cover_universal_2asset`), single-path, no external data dependency.
- `universal_portfolio/data.py` — Phase 1: NYSE(O) dataset loader + the cumulative→period conversion.
- `universal_portfolio/simulate.py` — Phase 2: correlated-GBM path generator; Phase 3: `Regime` + `simulate_regime_switching_gbm` for piecewise-constant-parameter paths.
- `universal_portfolio/strategies.py` — Phase 2: the same algorithms as `cover.py`, vectorized across simulation paths (log-wealth space throughout, for numerical stability at 20-year/60%-vol horizons); cross-checked against `cover.py` in tests. Phase 3: `*_path` variants that expose the full trajectory (not just final wealth) + `max_drawdown_from_log_wealth_path`. Phase 4: `cost_bps`/`rebalance_every` on the fixed-CRP and Universal Portfolio path functions.
- `universal_portfolio/costs.py` — Phase 4: sourced commission + effective-spread assumptions by vendor and liquidity tier.
- `universal_portfolio/experiments.py` — Phases 2-4: the experiments above, as pure functions returning DataFrames.
- `universal_portfolio/plotting.py` — chart rendering.
- `scripts/replicate_cover1991.py`, `scripts/mechanism_simulation.py` — CLI entry points.
- `tests/test_cover_replication.py` — Phase 1 tests: exact-number replication, the `universal wealth == mean(CRP wealth)` algebraic identity (true by construction, so it's what actually catches indexing/look-ahead bugs), a cross-check against `universal-portfolios`' own BCRP optimizer.
- `tests/test_strategies.py` — Phase 2-4 tests: batched implementation vs. Phase 1's single-path reference, the same algebraic identity batched, `BCRP ≥ Universal Portfolio` always (also algebraic, not empirical), GBM simulator calibration (including per-regime calibration), drawdown correctness on hand-constructed paths, and Phase 4's cost/frequency mechanics (including a caught-by-testing subtlety: rebalancing frequency changes the underlying wealth process even at zero cost, since the weight drifts between rebalances — cost and "structural" frequency effects had to be tested separately, not conflated).

## Planned next phases

Not yet implemented:

1. **Crisis-regime cost interaction** — combine Phase 3's spread-widening-in-a-crisis intuition with Phase 4's cost model quantitatively, rather than as a qualitative caveat.
2. **Modern out-of-sample** — real ETF pairs. Important caveat given Phases 2-3's results: candidates must be genuinely weakly/negatively correlated (e.g. equity+duration or equity+gold), not two single-name growth stocks, and *especially* not two tickers both carrying US-equity-market beta — correlation between those tends toward 1 exactly when it matters most (drawdowns, per Phase 3), which is precisely where Phase 3 shows rebalancing gets hurt worst.
