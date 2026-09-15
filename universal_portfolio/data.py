"""Loader for the classic NYSE(O) online-portfolio-selection benchmark.

Sourced from the `universal-portfolios` PyPI package (Marigold), which
bundles Cover's original NYSE dataset (36 stocks, 5651 trading days,
1962-07-03 to 1984-12-31) as `universal/data/nyse_o.csv`.

File format, determined empirically (the package ships no column->ticker
mapping and no format docstring for this file): each column is a
CUMULATIVE price relative indexed to day 0 (S_t = P_t / P_0), not a
day-over-day ratio. Naively multiplying a column's raw values together
(as if they were already period returns) compounds already-compounded
numbers and produces nonsense (some columns "grow" by e^3000+ over the
period). Taking a column's LAST raw value directly, instead, gives its
true total buy-and-hold wealth multiple.

That check is what identified columns "T" and "W": their last values are
8.915108 and 4.127591, matching Cover (1991) Table 8.1's Iroquois Brands
(8.9151x) and Kin Ark (4.1276x) reference figures to 5 significant figures.
`load_iroquois_kin_ark()` reconstructs the day-over-day series these two
columns imply (x_t = S_t / S_{t-1}, S_0 = 1) so it can be fed to
`cover.cover_universal_2asset`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

IROQUOIS_BRANDS_COLUMN = "T"
KIN_ARK_COLUMN = "W"


def load_nyse_o_price_relatives(columns: list[str] | None = None) -> pd.DataFrame:
    """Day-over-day gross price relatives for the NYSE(O) dataset.

    :param columns: subset of the 36 raw column labels to load (e.g.
        ["T", "W"]). Defaults to all 36.
    :returns: DataFrame of x_t = S_t / S_{t-1} with S_0 = 1 — i.e. values
        ready to feed into a constant-rebalanced-portfolio algorithm.
    """
    from universal import tools  # heavy optional dependency, imported lazily

    cum = tools.dataset("nyse_o")
    if columns is not None:
        cum = cum[columns]

    day0 = pd.DataFrame([np.ones(cum.shape[1])], columns=cum.columns)
    cum_with_day0 = pd.concat([day0, cum], ignore_index=True)
    return (cum_with_day0 / cum_with_day0.shift(1)).dropna().reset_index(drop=True)


def load_iroquois_kin_ark() -> pd.DataFrame:
    """The exact 2-stock series behind Cover (1991) Section 8's example."""
    return load_nyse_o_price_relatives([IROQUOIS_BRANDS_COLUMN, KIN_ARK_COLUMN])
