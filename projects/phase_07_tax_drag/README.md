# Phase 7 — tax drag

[Project overview](../../README.md) · [Introduction](../01_introduction/README.md) · [Previous: Phase 6](../phase_06_crisis_costs/README.md) · [Next: Phase 8](../phase_08_rolling_windows/README.md)

Status: complete. Run any commands below from the repository root.

Flagged since Phase 4 as likely the single largest unmodeled real cost.
The premise behind that flag is worth stating precisely, since "if I
don't cash out, is there a tax event?" conflates two different rules:

- **Wash sale (IRC §1091)** disallows (defers) a *loss* deduction when
  you sell at a loss and buy the *same or substantially identical*
  security within 30 days. It never touches gains, and a two-leg
  rebalance buys a genuinely *different* asset with the proceeds — so
  wash sale doesn't apply here in the way the question assumes, on
  either side of the trade.
- **Realization (IRC §1001)** is what actually governs: every *sale*
  realizes gain/loss in that tax year, whether the proceeds are
  withdrawn or immediately reinvested in something else. "Not cashing
  out" is irrelevant in a taxable account — the sale is what matters,
  not what happens to the cash afterward.

The account *wrapper* is what actually determines this: a taxable
brokerage account taxes every rebalancing sale per the above; a
tax-advantaged account (Traditional/Roth IRA, 401(k)) doesn't tax trades
inside it at all, until distribution. That's the scenario "no tax if I
stay invested" is correctly describing — it just requires the right
wrapper, not merely reinvesting.

`strategies.fixed_crp_log_wealth_path_with_tax` adds average-cost-basis
gain/loss tracking: on each rebalance, whichever leg is overweight is
sold down to target, realizing gain/loss on the sold fraction, taxed at
a scenario rate and funded by shrinking the whole portfolio (not by
missing the target weight). `tax_drag_sweep` crosses account type/
bracket against rebalancing frequency, `n_paths=5,000`, same (μ,σ,ρ) as
Phase 3/6's calm regime:

| Bracket | Frequency | Short-term rate drag | Long-term rate drag |
|---|---|---|---|
| Moderate (24% / 15% LTCG) | daily | 1.31 pp/yr | 0.82 pp/yr |
| Moderate | monthly | 1.02 pp/yr | 0.64 pp/yr |
| High (37%+NIIT / 20%+NIIT) | daily | 2.24 pp/yr | 1.30 pp/yr |
| High | monthly | 1.73 pp/yr | 1.01 pp/yr |

(drag = tax-advantaged growth minus the taxable scenario's; **percentage
points**, not basis points — note the unit change from Phases 4 and 6)

![Phase 7: annualized tax drag by bracket, holding-period assumption, and rebalancing frequency, synthetic symmetric-drift data](../../docs/images/phase7_tax_drag.png)

**This is an order of magnitude larger than every other friction this
repo has modeled.** Phase 4's bid-ask spread cost daily rebalancing at
most ~5bps/yr (single-stock tier, no price improvement); Phase 6's
crisis-cost-widening added at most another quarter of a bp. Tax drag here
runs 64-224 **basis points** per year depending on bracket, holding-
period assumption, and frequency — 15-40x larger than the spread cost
that got most of the earlier attention. The flag from Phase 4 was
correctly placed.

Monthly rebalancing recovers a meaningfully larger share of this drag
than it did for spread costs (Phase 4) or crisis-widened spread costs
(Phase 6) — because tax scales with the SIZE of realized gains, and
gains compound between rebalances the same way turnover does, so
trading less often defers more of both. It's still real drag even
monthly, though: this doesn't go away at low frequency, only shrinks.

**What this simplifies, stated plainly**: average-cost-basis accounting
(IRS-permitted for some securities, not universal) rather than exact
lot-by-lot FIFO/specific-ID; short/long-term is a *scenario* rate applied
to every realized gain, not derived from actually tracking each lot's
holding period (intractable for a portfolio trading on every rebalance —
short-term is the realistic case for daily/weekly rebalancing, long-term
an optimistic bound more plausible at low frequency); a realized loss
gives an immediate full tax rebate, assuming complete offset against
other income/gains that year, versus the real $3,000/yr ordinary-income
offset cap with carryforward; state tax (e.g. up to ~13.3% top marginal
in CA) isn't included and would stack on top of every number above.
