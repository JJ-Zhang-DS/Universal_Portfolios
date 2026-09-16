"""Static chart rendering for the Phase 2 experiments.

Style follows the project's data-viz conventions: fixed categorical color
order (never cycled/reassigned), one y-axis per panel (no dual-axis),
2px lines with >=8px end markers, recessive hairline gridlines, direct
end-of-line labels in ink (never colored text) so identity never depends
on color alone, legend present whenever a panel has >=2 series.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# fixed categorical order — see references/palette.md; never reassign per-chart
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)


def _place_end_labels(ax, entries, min_gap_frac: float = 0.06) -> None:
    """Place an end-marker at each (x, y) data point plus a text label.

    Labels that would collide (within `min_gap_frac` of the y-axis span)
    are nudged apart and connected back to their true point with a thin
    leader line, per the "don't stack converging end-labels" rule — a
    plain vertical offset detaches a label from its line and reads as
    noise, so we keep the visual link explicit instead.

    :param entries: list of (x, y, text, color).
    """
    for x, y, _, color in entries:
        ax.plot([x], [y], marker="o", markersize=8, color=color, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)

    ymin, ymax = ax.get_ylim()
    min_gap = min_gap_frac * (ymax - ymin)

    order = sorted(range(len(entries)), key=lambda i: entries[i][1])
    label_y = [entries[i][1] for i in order]
    for k in range(1, len(label_y)):
        if label_y[k] - label_y[k - 1] < min_gap:
            label_y[k] = label_y[k - 1] + min_gap

    for rank, idx in enumerate(order):
        x, y, text, color = entries[idx]
        ly = label_y[rank]
        if abs(ly - y) > 1e-9:
            ax.plot([x, x], [y, ly], color=BASELINE, linewidth=0.75, zorder=4)
        ax.annotate(
            text, xy=(x, ly), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK_PRIMARY
        )


def plot_rebalancing_premium(df: pd.DataFrame, path: str) -> None:
    """Two panels sharing x=rho: (1) empirical excess growth vs. the
    average of the two buy-and-hold legs, against the closed-form theory
    curve, per sigma; (2) probability that 50/50 rebalancing actually
    beats the BETTER leg — the practically relevant, much harder bar.
    """
    sigmas = sorted(df["sigma"].unique())
    colors = [BLUE, ORANGE, AQUA, YELLOW][: len(sigmas)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=SURFACE)

    ax1_entries, ax2_entries = [], []
    for sigma, color in zip(sigmas, colors):
        sub = df[df["sigma"] == sigma].sort_values("rho")
        ax1.plot(sub["rho"], sub["excess_vs_avg_leg_mean"], color=color, linewidth=2, zorder=3)
        ax1.plot(sub["rho"], sub["theory_excess"], color=color, linewidth=1, linestyle=(0, (3, 2)), zorder=2)
        ax1_entries.append((sub["rho"].iloc[-1], sub["excess_vs_avg_leg_mean"].iloc[-1], f"σ={sigma:.0%}", color))

        ax2.plot(
            sub["rho"], sub["prob_beats_best_leg"], color=color, linewidth=2, marker="o", markersize=5, zorder=3
        )
        ax2_entries.append((sub["rho"].iloc[-1], sub["prob_beats_best_leg"].iloc[-1], f"σ={sigma:.0%}", color))

    _place_end_labels(ax1, ax1_entries)
    _place_end_labels(ax2, ax2_entries)

    ax1.axhline(0, color=BASELINE, linewidth=1)
    ax1.set_title(
        "Rebalancing premium vs. average of the two legs\n(solid = simulated, dashed = ¼σ²(1−ρ) theory)",
        fontsize=10,
        color=INK_PRIMARY,
        loc="left",
    )
    ax1.set_xlabel("correlation ρ", fontsize=9, color=INK_SECONDARY)
    ax1.set_ylabel("annualized excess log-growth", fontsize=9, color=INK_SECONDARY)

    ax2.axhline(0.5, color=BASELINE, linewidth=1)
    ax2.set_title(
        "P(50/50 CRP beats the BETTER leg)\n— the bar that actually matters to an investor",
        fontsize=10,
        color=INK_PRIMARY,
        loc="left",
    )
    ax2.set_xlabel("correlation ρ", fontsize=9, color=INK_SECONDARY)
    ax2.set_ylabel("probability", fontsize=9, color=INK_SECONDARY)
    ax2.set_ylim(0, 1)

    for ax in (ax1, ax2):
        _style_axes(ax)
        ax.set_xlim(-1.0, 1.25)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_drift_difference(df: pd.DataFrame, path: str) -> None:
    """Single panel, x = drift gap between the two assets: winner/loser
    buy-and-hold, fixed 50/50 CRP, BCRP (hindsight), Universal Portfolio.
    Where the 50/50 line crosses below the winner's line is the direct
    falsification point for "just rebalance two volatile assets"."""
    df = df.sort_values("delta_mu")
    series = [
        ("winner_growth", "winner (buy & hold)", GREEN),
        ("loser_growth", "loser (buy & hold)", RED),
        ("fixed5050_growth", "fixed 50/50 CRP", BLUE),
        ("bcrp_growth", "BCRP (hindsight, not tradeable)", VIOLET),
        ("universal_growth", "Universal Portfolio", ORANGE),
    ]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    entries = []
    for col, label, color in series:
        style = dict(linewidth=2, zorder=3)
        if col == "bcrp_growth":
            style.update(linestyle=(0, (3, 2)), linewidth=1.5)
        ax.plot(df["delta_mu"], df[col], color=color, **style)
        entries.append((df["delta_mu"].iloc[-1], df[col].iloc[-1], label, color))
    _place_end_labels(ax, entries)

    ax.set_title(
        "Fixed-weight rebalancing vs. drift gap between the two assets\n"
        "(σ=40% both legs, ρ=0 — falsifies 'just rebalance any two volatile stocks')",
        fontsize=10,
        color=INK_PRIMARY,
        loc="left",
    )
    ax.set_xlabel("drift gap Δμ (annualized, asset 1 − asset 2)", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("annualized log-growth rate", fontsize=9, color=INK_SECONDARY)
    ax.set_xlim(df["delta_mu"].min(), df["delta_mu"].max() * 1.55)
    _style_axes(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_horizon_convergence(df: pd.DataFrame, path: str) -> None:
    """Single series: Cover's regret R_T = (BCRP − Universal Portfolio)
    log-wealth per day, vs. horizon. Should shrink toward 0 — that's the
    actual (asymptotic) claim the paper makes, not a finite-sample win."""
    df = df.sort_values("horizon_years")

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    ax.plot(df["horizon_years"], df["mean_regret_per_day"], color=BLUE, linewidth=2, marker="o", markersize=8,
             markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)

    for _, row in df.iterrows():
        ax.annotate(
            f"{row['mean_regret_per_day']:.4f}",
            xy=(row["horizon_years"], row["mean_regret_per_day"]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=INK_SECONDARY,
        )

    ax.set_title(
        "Cover's regret shrinks with horizon (it never claims to win in finite samples)",
        fontsize=10,
        color=INK_PRIMARY,
        loc="left",
    )
    ax.set_xlabel("horizon (years)", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("mean regret per day: (BCRP − UP) log-wealth / T", fontsize=9, color=INK_SECONDARY)
    ax.set_xticks(df["horizon_years"])
    _style_axes(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_regime_shift(df: pd.DataFrame, path: str) -> None:
    """Grouped bars: the crisis window's cost (crisis minus its no-crisis
    counterfactual) to fixed 50/50 CRP vs. plain buy-and-hold, per crisis
    definition. Bars rather than lines because the x-axis (scenario) is
    categorical, not ordered/continuous.
    """
    scenarios = ["correlation_only", "volatility_only", "joint_crisis"]
    scenario_labels = {
        "correlation_only": "correlation\nonly (ρ→0.95)",
        "volatility_only": "volatility\nonly (σ→55%)",
        "joint_crisis": "joint crisis\n(ρ,σ↑, μ<0)",
    }

    crisis = df[df["regime"] == "crisis"].set_index("scenario").loc[scenarios]
    calm = df[df["regime"] == "no_crisis"].set_index("scenario").loc[scenarios]

    panels = [
        (
            crisis["crp_max_drawdown"] - calm["crp_max_drawdown"],
            crisis["bh_leg_avg_max_drawdown"] - calm["bh_leg_avg_max_drawdown"],
            "Extra drawdown from the crisis window\n(crisis − no-crisis; more negative = worse)",
            "Δ max drawdown",
        ),
        (
            crisis["crp_final_growth"] - calm["crp_final_growth"],
            crisis["bh_leg_avg_final_growth"] - calm["bh_leg_avg_final_growth"],
            "Growth-rate cost of the crisis window\n(crisis − no-crisis)",
            "Δ annualized log-growth",
        ),
    ]

    x = np.arange(len(scenarios))
    width = 0.32
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURFACE)

    for ax, (delta_crp, delta_bh, title, ylabel) in zip(axes, panels):
        ax.bar(x - width / 2, delta_crp.values, width, color=BLUE, zorder=3)
        ax.bar(x + width / 2, delta_bh.values, width, color=ORANGE, zorder=3)

        for xi, v in zip(x - width / 2, delta_crp.values):
            ax.annotate(f"{v:+.1%}", xy=(xi, v), xytext=(0, 3 if v >= 0 else -12),
                        textcoords="offset points", ha="center", fontsize=8, color=INK_SECONDARY)
        for xi, v in zip(x + width / 2, delta_bh.values):
            ax.annotate(f"{v:+.1%}", xy=(xi, v), xytext=(0, 3 if v >= 0 else -12),
                        textcoords="offset points", ha="center", fontsize=8, color=INK_SECONDARY)

        ax.axhline(0, color=BASELINE, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([scenario_labels[s] for s in scenarios], fontsize=9, color=INK_SECONDARY)
        ax.set_title(title, fontsize=10, color=INK_PRIMARY, loc="left")
        ax.set_ylabel(ylabel, fontsize=9, color=INK_SECONDARY)
        _style_axes(ax)

    handles = [plt.Line2D([0], [0], color=BLUE, lw=6), plt.Line2D([0], [0], color=ORANGE, lw=6)]
    fig.legend(
        handles, ["fixed 50/50 CRP", "buy & hold (leg avg)"], loc="upper center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, 1.06), fontsize=9, labelcolor=INK_SECONDARY,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_transaction_costs(df_etf: pd.DataFrame, df_stock: pd.DataFrame, path: str) -> None:
    """Grouped bars: cost drag (vs. frictionless daily fixed 50/50) by
    rebalancing frequency, for the Fidelity vs. no-price-improvement
    vendor scenarios, one panel per liquidity tier. Bars because the
    x-axis (frequency) is categorical/ordinal, not continuous.
    """
    freq_order = ["daily", "weekly", "monthly"]

    def drag_table(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        crp = df[df["strategy"] == "fixed_50_50"].set_index(["cost_scenario", "frequency"])
        ref = crp.loc[("frictionless", "daily"), "annualized_growth"]
        fidelity = np.array([ref - crp.loc[("fidelity", f), "annualized_growth"] for f in freq_order])
        no_pi = np.array([ref - crp.loc[("no_price_improvement", f), "annualized_growth"] for f in freq_order])
        return fidelity, no_pi

    x = np.arange(len(freq_order))
    width = 0.32
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURFACE, sharey=False)

    for ax, df, title in zip(axes, (df_etf, df_stock), ("mega-liquid ETF tier (SPY/QQQ-like)", "single-stock tier")):
        fidelity, no_pi = drag_table(df)

        ax.bar(x - width / 2, fidelity * 1e4, width, color=BLUE, zorder=3)
        ax.bar(x + width / 2, no_pi * 1e4, width, color=ORANGE, zorder=3)

        for xi, v in zip(x - width / 2, fidelity * 1e4):
            ax.annotate(f"{v:.1f}", xy=(xi, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK_SECONDARY)
        for xi, v in zip(x + width / 2, no_pi * 1e4):
            ax.annotate(f"{v:.1f}", xy=(xi, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK_SECONDARY)

        ax.set_xticks(x)
        ax.set_xticklabels(freq_order, fontsize=9, color=INK_SECONDARY)
        ax.set_title(title, fontsize=10, color=INK_PRIMARY, loc="left")
        ax.set_ylabel("cost drag vs. frictionless daily (bps/yr)", fontsize=9, color=INK_SECONDARY)
        ax.set_ylim(bottom=0)
        _style_axes(ax)

    fig.suptitle(
        "Fixed 50/50 CRP: growth given up to costs, by vendor scenario and rebalancing frequency",
        fontsize=10.5, color=INK_PRIMARY, x=0.01, ha="left", y=1.04,
    )
    handles = [plt.Line2D([0], [0], color=BLUE, lw=6), plt.Line2D([0], [0], color=ORANGE, lw=6)]
    fig.legend(
        handles, ["Fidelity (disclosed effective spread)", "no price improvement (illustrative)"],
        loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.99), fontsize=9, labelcolor=INK_SECONDARY,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_real_pair_results(df: pd.DataFrame, path: str) -> None:
    """Two panels, both x=pair (sorted by realized correlation, low to
    high): (1) actual vs. theory excess growth vs. the average leg --
    does Phase 2's closed-form formula survive contact with real,
    non-lognormal, non-stationary market data; (2) excess vs. the BETTER
    leg -- the practically relevant, much harder bar, on real pairs
    instead of synthetic ones.
    """
    df = df.sort_values("realized_rho")
    pairs = df["pair"].tolist()
    x = np.arange(len(pairs))
    width = 0.32

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5), facecolor=SURFACE)

    ax1.bar(x - width / 2, df["excess_vs_avg_leg"] * 1e4, width, color=BLUE, zorder=3)
    ax1.bar(x + width / 2, df["theory_excess_vs_avg_leg"] * 1e4, width, color=ORANGE, zorder=3)
    ax1.axhline(0, color=BASELINE, linewidth=1)
    ax1.set_title(
        "Actual vs. theory excess growth (vs. average leg)\nsolid = realized 2016-2026, orange = ¼σ²(1−ρ) prediction",
        fontsize=10, color=INK_PRIMARY, loc="left",
    )
    ax1.set_ylabel("annualized excess (bps)", fontsize=9, color=INK_SECONDARY)
    handles1 = [plt.Line2D([0], [0], color=BLUE, lw=6), plt.Line2D([0], [0], color=ORANGE, lw=6)]
    ax1.legend(handles1, ["realized", "theory"], loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)

    ax2.bar(x, df["excess_vs_better_leg"] * 1e4, width * 1.6, color=BLUE, zorder=3)
    ax2.axhline(0, color=BASELINE, linewidth=1)
    ax2.set_title(
        "Excess vs. the BETTER leg\n(the bar an investor actually faces, not the average)",
        fontsize=10, color=INK_PRIMARY, loc="left",
    )
    ax2.set_ylabel("annualized excess (bps)", fontsize=9, color=INK_SECONDARY)

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(pairs, fontsize=8.5, color=INK_SECONDARY, rotation=30, ha="right")
        _style_axes(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_rolling_correlation(series: dict[str, "pd.Series"], path: str, highlight: list[tuple] = ()) -> None:
    """Rolling realized correlation over time for one or more pairs, with
    shaded spans over named historical windows (e.g. the 2020 COVID crash,
    the 2022 rate-hike selloff) -- the real-data counterpart to Phase 3's
    synthetic correlation regime shift.

    :param series: {label: pandas Series of rolling correlation, DatetimeIndex}.
    :param highlight: list of (start, end, label) to shade.
    """
    colors = [BLUE, ORANGE, AQUA, YELLOW]
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)

    for (start, end, label) in highlight:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=GRIDLINE, zorder=1)
        ax.annotate(
            label, xy=(pd.Timestamp(start), 1.0), xytext=(3, -3), textcoords="offset points",
            fontsize=8, color=INK_MUTED, va="top",
        )

    entries = []
    for (label, s), color in zip(series.items(), colors):
        s = s.dropna()
        ax.plot(s.index, s.values, color=color, linewidth=1.5, zorder=3)
        entries.append((s.index[-1], s.values[-1], label, color))

    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.set_ylim(-1.0, 1.05)
    ax.set_title("60-day rolling realized correlation", fontsize=10, color=INK_PRIMARY, loc="left")
    ax.set_ylabel("correlation", fontsize=9, color=INK_SECONDARY)
    _style_axes(ax)

    handles = [plt.Line2D([0], [0], color=c, lw=2) for _, c in zip(series, colors)]
    ax.legend(handles, list(series.keys()), loc="lower left", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_crisis_cost_interaction(df: pd.DataFrame, path: str) -> None:
    """Grouped bars: how much Phase 4's naive constant-cost assumption
    understates the true cost once the spread is allowed to widen during
    Phase 3's crisis window, by rebalancing frequency and liquidity tier.
    """
    freq_order = ["daily", "weekly", "monthly"]
    etf = df[df["tier"] == "mega_liquid_etf"].set_index("frequency").loc[freq_order, "naive_understatement_bps"]
    stock = df[df["tier"] == "single_stock"].set_index("frequency").loc[freq_order, "naive_understatement_bps"]

    x = np.arange(len(freq_order))
    width = 0.32
    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=SURFACE)

    ax.bar(x - width / 2, etf.values, width, color=BLUE, zorder=3)
    ax.bar(x + width / 2, stock.values, width, color=ORANGE, zorder=3)

    for xi, v in zip(x - width / 2, etf.values):
        ax.annotate(f"{v:.2f}", xy=(xi, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK_SECONDARY)
    for xi, v in zip(x + width / 2, stock.values):
        ax.annotate(f"{v:.2f}", xy=(xi, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK_SECONDARY)

    ax.set_xticks(x)
    ax.set_xticklabels(freq_order, fontsize=9, color=INK_SECONDARY)
    ax.set_title(
        "Cost UNDERSTATEMENT from assuming calm-regime spread\nthroughout a crisis window (vs. 3x wider during it)",
        fontsize=10, color=INK_PRIMARY, loc="left",
    )
    ax.set_ylabel("naive understatement (bps/yr)", fontsize=9, color=INK_SECONDARY)
    _style_axes(ax)

    handles = [plt.Line2D([0], [0], color=BLUE, lw=6), plt.Line2D([0], [0], color=ORANGE, lw=6)]
    ax.legend(handles, ["mega-liquid ETF", "single stock"], loc="upper right", frameon=False,
              fontsize=9, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_crisis_multiplier_sensitivity(df: pd.DataFrame, path: str) -> None:
    """Line chart: does the (stylized, not fitted) crisis spread-widening
    multiplier itself matter much -- swept 1x (no widening) to 10x, at
    daily rebalancing, both liquidity tiers.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=SURFACE)
    entries = []

    for tier, color in [("mega_liquid_etf", BLUE), ("single_stock", ORANGE)]:
        sub = df[df["tier"] == tier].sort_values("crisis_multiplier")
        ax.plot(sub["crisis_multiplier"], sub["drag_vs_frictionless_bps"], color=color, linewidth=2,
                marker="o", markersize=6, zorder=3)
        entries.append((sub["crisis_multiplier"].iloc[-1], sub["drag_vs_frictionless_bps"].iloc[-1], tier, color))

    _place_end_labels(ax, [(x, y, {"mega_liquid_etf": "mega-liquid ETF", "single_stock": "single stock"}[t], c) for x, y, t, c in entries])

    ax.set_title(
        "Total cost drag vs. the crisis spread-widening multiplier\n(1x = no widening at all; 3x is this repo's base assumption)",
        fontsize=10, color=INK_PRIMARY, loc="left",
    )
    ax.set_xlabel("crisis spread-widening multiplier", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("drag vs. frictionless (bps/yr)", fontsize=9, color=INK_SECONDARY)
    ax.set_xlim(0.5, 12)
    _style_axes(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_tax_drag(df: pd.DataFrame, path: str) -> None:
    """Two panels (moderate / high bracket), each x=frequency with
    short-term vs. long-term rate as grouped bars: tax drag vs. the
    tax-advantaged (IRA/401k) baseline. Scale here is percentage points,
    not bps -- an order of magnitude larger than Phase 4's spread-cost
    drag or Phase 6's crisis-cost-widening drag.
    """
    freq_order = ["daily", "weekly", "monthly"]
    advantaged = df[df["scenario"].str.startswith("tax-advantaged")].set_index("frequency")["annualized_growth"]

    def drag_for(prefix: str) -> tuple:
        short = df[df["scenario"] == f"{prefix}, short-term"].set_index("frequency")["annualized_growth"]
        long = df[df["scenario"] == f"{prefix}, long-term"].set_index("frequency")["annualized_growth"]
        return (advantaged - short).loc[freq_order], (advantaged - long).loc[freq_order]

    moderate_short, moderate_long = drag_for("moderate bracket (24% ordinary / 15% LTCG)")
    high_short, high_long = drag_for("high bracket (37% ordinary + 3.8% NIIT / 20% LTCG + 3.8% NIIT)")

    x = np.arange(len(freq_order))
    width = 0.32
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), facecolor=SURFACE, sharey=True)

    for ax, short, long, title in [
        (ax1, moderate_short, moderate_long, "Moderate bracket\n(24% ordinary / 15% LTCG)"),
        (ax2, high_short, high_long, "High bracket\n(37%+NIIT ordinary / 20%+NIIT LTCG)"),
    ]:
        ax.bar(x - width / 2, short.values * 100, width, color=BLUE, zorder=3)
        ax.bar(x + width / 2, long.values * 100, width, color=ORANGE, zorder=3)
        for xi, v in zip(x - width / 2, short.values * 100):
            ax.annotate(f"{v:.2f}", xy=(xi, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK_SECONDARY)
        for xi, v in zip(x + width / 2, long.values * 100):
            ax.annotate(f"{v:.2f}", xy=(xi, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK_SECONDARY)
        ax.set_xticks(x)
        ax.set_xticklabels(freq_order, fontsize=9, color=INK_SECONDARY)
        ax.set_title(title, fontsize=10, color=INK_PRIMARY, loc="left")
        _style_axes(ax)

    ax1.set_ylabel("tax drag vs. tax-advantaged account (pp/yr)", fontsize=9, color=INK_SECONDARY)
    handles = [plt.Line2D([0], [0], color=BLUE, lw=6), plt.Line2D([0], [0], color=ORANGE, lw=6)]
    ax1.legend(handles, ["short-term rate", "long-term rate"], loc="upper right", frameon=False,
               fontsize=9, labelcolor=INK_SECONDARY)

    fig.suptitle(
        "Taxable-account drag vs. an IRA/401(k), fixed 50/50 CRP",
        fontsize=11, color=INK_PRIMARY, x=0.01, ha="left", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
