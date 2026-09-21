# Universal Portfolio: Introduction

This subproject introduces Thomas Cover's Universal Portfolio (UP) before the repository moves into replication, simulation, market frictions, and real-data tests.

> **Core idea:** today's allocation is the historical-wealth-weighted average of all candidate constant-rebalanced portfolios.

UP is a **model-free, multi-asset allocation algorithm**. It does not predict returns or train on features. Instead, it evaluates many fixed-weight strategies and gives more influence to those that have accumulated more wealth so far.

## 1. Constant-rebalanced portfolios (CRPs)

For two assets, let $b\in[0,1]$ be the allocation to asset A; $1-b$ goes to asset B. Each value of $b$ defines a **constant-rebalanced portfolio (CRP)**.

| $b$ | Allocation |
|---:|---|
| 0.00 | 0% A / 100% B |
| 0.25 | 25% A / 75% B |
| 0.50 | 50% A / 50% B |
| 0.75 | 75% A / 25% B |
| 1.00 | 100% A / 0% B |

A CRP restores its target weights after every period. If period-$t$ gross returns are $x_{t,A}$ and $x_{t,B}$, candidate $b$'s wealth evolves as

$$
S_t(b)=S_{t-1}(b)\left[b x_{t,A}+(1-b)x_{t,B}\right],
\qquad S_0(b)=1.
$$

## 2. Historical-wealth weighting

UP does not simply choose yesterday's winning candidate. It combines **all** candidates, with influence proportional to the wealth each has accumulated through the previous period.

For a discrete grid $b_1,\ldots,b_m$,

$$
p_t(b_j)=\frac{S_{t-1}(b_j)}
{\sum_{k=1}^{m}S_{t-1}(b_k)}
$$

and today's allocation to A is

$$
\hat b_t=\sum_{j=1}^{m}p_t(b_j)b_j
=
\frac{\sum_{j=1}^{m} b_j S_{t-1}(b_j)}
{\sum_{j=1}^{m} S_{t-1}(b_j)}.
$$

### Small example

Suppose five candidate CRPs have this wealth at the end of yesterday:

| A weight $b$ | Wealth $S_{t-1}(b)$ | Wealth share |
|---:|---:|---:|
| 0.00 | 1.0 | 10% |
| 0.25 | 1.5 | 15% |
| 0.50 | 2.0 | 20% |
| 0.75 | 3.0 | 30% |
| 1.00 | 2.5 | 25% |

Their total wealth is 10, so

$$
\hat b_t
=0(0.10)+0.25(0.15)+0.50(0.20)+0.75(0.30)+1(0.25)
=0.6125.
$$

UP therefore holds **61.25% A and 38.75% B**. Historically successful CRPs have more influence, but the process is not winner-takes-all.

## 3. Cover's continuous formulation

Cover's original two-asset formulation averages over every $b\in[0,1]$, not only a finite grid:

$$
\hat b_t=
\frac{\int_0^1 b\,S_{t-1}(b)\,db}
{\int_0^1 S_{t-1}(b)\,db}.
$$

Because every candidate begins with the same wealth, $S_0(b)=1$, the first allocation is

$$
\hat b_1=
\frac{\int_0^1 b\,db}{\int_0^1 1\,db}
=\frac12.
$$

Thus two-asset UP starts at 50/50 and adapts as candidate CRPs accumulate different wealth.

For more than two assets, $b$ becomes a vector on the portfolio simplex. The same wealth-weighting principle applies.

## 4. CRP, BCRP, and UP are different

| Term | Meaning | Uses future information? |
|---|---|---|
| **CRP** | One fixed-weight, periodically rebalanced strategy | No, if chosen in advance |
| **BCRP** | The best CRP found after searching all fixed weights over the complete sample | **Yes—hindsight benchmark** |
| **Universal Portfolio** | Online wealth-weighted mixture of all CRPs, using information only through yesterday | No |

The **best constant-rebalanced portfolio (BCRP)** is an oracle benchmark, not an investable historical strategy unless its optimal weight was known before the sample began.

Cover proved that UP asymptotically approaches BCRP's growth rate under broad conditions. This is an asymptotic regret guarantee—not a promise to beat the best individual asset in every finite sample.

## 5. The 73.62× versus 38.67× distinction

This repository reproduces Cover's Iroquois Brands / Kin Ark example from 1962–1984:

| Strategy | Terminal wealth |
|---|---:|
| Iroquois Brands buy-and-hold | 8.9151× |
| Kin Ark buy-and-hold | 4.1276× |
| BCRP, found in hindsight (55% Iroquois) | 73.6190× |
| Universal Portfolio, online without look-ahead | 38.6727× |

The frequently repeated **73.62×** belongs to BCRP: its optimal fixed weight was identified after observing the entire 22-year path. The feasible online Universal Portfolio achieved **38.67×**.

That difference is central to the paper: it measures how closely an online strategy can track an infeasible hindsight oracle.

## 6. What this repository tests

After validating the reference replication, the project asks when the mechanism remains economically useful:

1. **Volatility:** does greater volatility create a larger rebalancing premium?
2. **Correlation:** does lower correlation help, and what happens when correlation rises in a crisis?
3. **Drift differences:** what happens when rebalancing repeatedly trims a persistent winner?
4. **Horizon:** how quickly does UP's per-period regret relative to BCRP shrink?
5. **Regime shifts:** how does calm-to-crisis-to-calm behavior differ from constant-parameter GBM?
6. **Transaction costs:** do spreads and rebalancing frequency erase the benefit?
7. **Taxes:** how much does repeated gain realization reduce after-tax growth?
8. **Real data and rolling windows:** do conclusions survive outside one model or one favorable start/end date?

A distinction used throughout the project is:

- **Beating the average of the two assets** is the classical rebalancing-premium comparison.
- **Beating the better asset** is a harder standard and often the economically relevant one.

The simulations report both.

## 7. Run the reference replication

~~~bash
pip install -r requirements.txt
python -m scripts.replicate_cover1991
pytest
~~~

Continue with the [project overview and phase-by-phase results](../../README.md).

## References

- Thomas M. Cover, “Universal Portfolios,” *Mathematical Finance* 1(1), 1991: [paper](https://isl.stanford.edu/~cover/papers/paper93.pdf)
- [Conversation that initiated this project](https://chatgpt.com/share/6aae9f03-5c14-83ea-8254-1f9c541b65db)
