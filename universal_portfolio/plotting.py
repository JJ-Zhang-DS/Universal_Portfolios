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
