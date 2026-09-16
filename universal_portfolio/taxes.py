"""US tax treatment of rebalancing, and why "I didn't cash out" doesn't
avoid it.

Two rules get conflated in practice, and neither does what the conflation
implies:

- **Realization (IRC §1001)**: every SALE of a security realizes its
  gain/loss in that tax year, whether or not the proceeds are withdrawn
  or immediately reinvested in something else. Reinvesting is irrelevant;
  the sale is what matters. In a rebalance between two different assets
  (sell some of the overweight leg, buy the underweight one), the sold
  leg's gain or loss is realized right there -- "staying invested" in a
  DIFFERENT asset changes nothing about that.
- **Wash sale (IRC §1091)**: disallows (defers into the replacement
  shares' basis) a LOSS deduction when you sell at a loss and buy the
  SAME or a "substantially identical" security within 30 days before or
  after. It never touches gains, and it doesn't apply to buying a
  DIFFERENT asset -- which is exactly what a two-leg rebalance does. It
  is not a general "reinvested = no tax" shield; if anything it's a
  narrow rule that can make a specific LOSS temporarily undeductible, the
  opposite of a tax break.

The account wrapper is what actually determines this, not whether you
cash out:
- **Taxable brokerage account**: every rebalancing sale is realized and
  taxed in that year, per above.
- **Tax-advantaged account (Traditional/Roth IRA, 401(k))**: trades
  inside the account are not taxable events at all, until distribution
  (and Roth qualified distributions aren't taxed even then). This is the
  scenario "if I don't cash out, no tax" is actually describing correctly
  -- it just requires being in one of these wrappers, not merely
  reinvesting the proceeds.

Rates sourced September 2026 (Tax Foundation / Kiplinger 2026 bracket
tables, post-OBBBA -- the 2017 TCJA individual rate structure, including
the 37% top ordinary bracket, was made permanent starting 2026):
short-term gains (held <=1yr) are taxed as ordinary income, brackets
10/12/22/24/32/35/37%; long-term gains (held >1yr) at 0/15/20%; the Net
Investment Income Tax (NIIT) adds 3.8% to both for MAGI above $200k
single / $250k married filing jointly.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TaxScenario:
    name: str
    short_term_rate: float  # ordinary-income bracket, gains on positions held <=1yr
    long_term_rate: float  # preferential LTCG bracket, gains on positions held >1yr


# Illustrative brackets, not a substitute for the investor's actual return.
# State tax (e.g. up to ~13.3% top marginal in CA) would stack on top of
# all of these and isn't included -- varies too much by state to assume.
TAX_ADVANTAGED = TaxScenario("tax-advantaged (IRA/401k)", short_term_rate=0.0, long_term_rate=0.0)
MODERATE_BRACKET = TaxScenario("moderate bracket (24% ordinary / 15% LTCG)", short_term_rate=0.24, long_term_rate=0.15)
HIGH_BRACKET = TaxScenario(
    "high bracket (37% ordinary + 3.8% NIIT / 20% LTCG + 3.8% NIIT)",
    short_term_rate=0.37 + 0.038,
    long_term_rate=0.20 + 0.038,
)
