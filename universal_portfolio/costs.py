"""Transaction-cost assumptions: vendor commission + effective bid-ask
spread. Sourced September 2026; see README's Phase 4 section for the
full citation trail.

The commission side is settled and not worth modeling as a variable:
Fidelity, Schwab, Vanguard, and every other major US broker have charged
$0 commission on online US stock/ETF trades since 2019
(fidelity.com/trading/commissions-margin-rates, checked Sep 2026).
Fidelity's narrow exception -- up to $100 (5% of trade, capped) on
purchases of ~120 niche ETFs whose issuers don't participate in its
revenue-sharing program, effective June 2026 -- does not apply to this
repo's tickers: SPY (SPDR), QQQ (Invesco), GLD (SPDR), TLT (iShares) are
all from providers explicitly named as unaffected.

So commission is not where "which vendor" or "ask vs. bid vs. market
price" actually bites -- the bid-ask spread is. Fidelity's own Q1 2026
execution-quality disclosure (fidelity.com/trading/execution-quality/
overview, SEC Rule 605 data, Apr 2025-Mar 2026) states: 95.37% of shares
receive price improvement over the NBBO, and average EFFECTIVE spread
(the distance from the NBBO midpoint to your actual execution price,
already net of that price improvement -- i.e. what you actually pay, not
the quoted spread) is $0.0056/share, against a $0.57-vs-$4.38-per-100-
shares industry-average-savings comparison Fidelity discloses on the same
page. That $0.0056 figure is a broad average across Fidelity's entire
Rule-605-eligible order flow, not broken out per ticker -- it is NOT
SPY-specific or QQQ-specific data.

The per-tier bps assumptions below are this repo's own, using that
figure and the well-documented fact that SPY/QQQ trade with penny-wide
NBBO quoted spreads (at ~$500-650/share, ~0.15-0.2bps) as anchors, not
literal per-name Fidelity statistics. They're deliberately rounded up
from those anchors to stay conservative. A "no_price_improvement" tier
stands in for a lower-execution-quality vendor (the full quoted NBBO
spread, none of Fidelity's disclosed price improvement) -- illustrative,
not a specific named competitor's verified number.

Regulatory pass-through fees (SEC fee on sells, FINRA TAF) are <0.05bps
at current rates and are not modeled -- immaterial next to any spread
tier here.
"""
from __future__ import annotations

TRADING_DAYS_PER_YEAR = 252

# one-way cost in bps of trade value, charged on turnover = |target_weight -
# drifted_weight| (see strategies.fixed_crp_log_wealth_path) -- NOT doubled
# for "round trip": a single rebalance simultaneously sells one leg and buys
# the other, each already paying its own half-spread, and those two paid
# half-spreads are what these numbers represent.
CALM_SPREAD_BPS = {
    ("mega_liquid_etf", "fidelity"): 0.3,
    ("mega_liquid_etf", "no_price_improvement"): 0.7,
    ("single_stock", "fidelity"): 2.0,
    ("single_stock", "no_price_improvement"): 4.0,
}

# stylized: spreads widen under stress as market makers demand more
# compensation for adverse-selection/inventory risk -- a well-documented
# market-microstructure phenomenon, not a multiplier fitted for this study.
CRISIS_SPREAD_MULTIPLIER = 3.0


def spread_cost_bps(tier: str, vendor: str, crisis: bool = False) -> float:
    """One-way turnover cost in bps for a (liquidity tier, vendor/execution-
    quality) pair, optionally under the Phase 3 crisis-regime multiplier.

    :param tier: "mega_liquid_etf" or "single_stock".
    :param vendor: "fidelity" or "no_price_improvement".
    """
    base = CALM_SPREAD_BPS[(tier, vendor)]
    return base * CRISIS_SPREAD_MULTIPLIER if crisis else base
