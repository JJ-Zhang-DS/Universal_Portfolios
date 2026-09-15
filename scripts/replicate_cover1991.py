"""Reproduce Cover (1991) Table 8.1: Iroquois Brands vs. Kin Ark.

    python -m scripts.replicate_cover1991

Prints buy-and-hold wealth for each stock, the Best Constant Rebalanced
Portfolio (BCRP, hindsight-optimal — NOT a tradeable strategy), and Cover's
actual Universal Portfolio (no look-ahead). See universal_portfolio/data.py
for where this data comes from and why it needs converting before use.
"""
from universal_portfolio.cover import cover_universal_2asset
from universal_portfolio.data import load_iroquois_kin_ark


def main() -> None:
    X_df = load_iroquois_kin_ark()
    X = X_df.to_numpy()

    iroquois_wealth, kin_ark_wealth = X.prod(axis=0)
    result = cover_universal_2asset(X)

    print(f"Trading days:                      {len(X)}")
    print(f"Iroquois Brands buy & hold:         {iroquois_wealth:.4f}x")
    print(f"Kin Ark buy & hold:                 {kin_ark_wealth:.4f}x")
    print(
        f"BCRP (hindsight optimum, "
        f"{result.best_crp_weight:.0%} Iroquois):    {result.bcrp_wealth:.3f}x"
    )
    print(f"Universal Portfolio (no look-ahead): {result.final_universal_wealth:.4f}x")
    print()
    print("Reference (Cover 1991, Table 8.1): 8.9151x / 4.1276x / 73.619x / 38.6727x")
    print()
    print(
        "Note: 73.619x is a hindsight-optimal fixed weight, unknowable in "
        "1962 — not a return any strategy could have earned. Cover's "
        "actual algorithm (no future information) reaches 38.6727x."
    )


if __name__ == "__main__":
    main()
