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

## Layout

- `universal_portfolio/cover.py` — the algorithm (`cover_universal_2asset`), no external data dependency.
- `universal_portfolio/data.py` — NYSE(O) dataset loader + the cumulative→period conversion above.
- `scripts/replicate_cover1991.py` — runs the replication end to end.
- `tests/test_cover_replication.py` — the exact-number replication, an algebraic invariant check (`universal wealth == mean(CRP wealth)`, true by construction for any input, so it's what actually catches indexing/look-ahead bugs), and a cross-check against `universal-portfolios`' own independent BCRP optimizer.

## Planned next phases

Not yet implemented — flagging so scope is explicit:

1. **Mechanism simulation** — correlated GBM Monte Carlo, sweeping volatility/correlation/drift-difference, to separate the rebalancing-premium effect (≈ ¼σ²(1-ρ) for the symmetric case) from Cover's online-learning mechanism.
2. **Failure-mode stress tests** — persistent winner/loser, correlation → 1, volatility regime shift. This is the direct falsification test for "just pick two volatile uncorrelated stocks and rebalance."
3. **Transaction costs** — proportional cost + rebalancing frequency (daily/weekly/monthly), since Cover's 1991 result is frictionless and daily rebalancing at 1960s-style turnover is not free today.
4. **Modern out-of-sample** — real ETF pairs, with an important caveat: candidates must be genuinely weakly/negatively correlated (e.g. equity+duration or equity+gold), not two single-name growth stocks, and *especially* not two tickers both carrying US-equity-market beta — correlation between those tends toward 1 exactly when it matters most (drawdowns), which is precisely the failure mode in (2).
