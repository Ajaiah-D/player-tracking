"""Block-height timeline: where each team's out-of-possession block sits
over the match, drawn as a band from the defensive line to the front
line with the centroid through it."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from soccervision.pitch import PITCH_LENGTH
from soccervision.shape import minute_trend

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
TEAM_COLORS = {"home": "#2a78d6", "away": "#eb6834"}


def plot_block_height(shape, out_path, goals=(), title="Out-of-possession block", smooth_min=5):
    """shape: labelled output of shape.label_phases. goals: iterable of
    (minute, period, scoring_team)."""
    shape = shape[shape["phase"] == "out_of_possession"]
    # running score (home-away) after each goal
    scorelines, tally = [], {"home": 0, "away": 0}
    for minute, period, scorer in sorted(goals, key=lambda g: (g[1], g[0])):
        tally[scorer] += 1
        scorelines.append((minute, period, scorer, f"{tally['home']}-{tally['away']}"))
    teams = ["home", "away"]
    periods = sorted(shape["period"].unique())
    half_len = [shape.loc[shape["period"] == p, "minute"].max() - 45 * (p - 1) for p in periods]

    fig, axes = plt.subplots(len(teams), len(periods), figsize=(12, 6.8), sharey=True,
                             gridspec_kw={"width_ratios": half_len, "wspace": 0.04, "hspace": 0.35},
                             facecolor=SURFACE, squeeze=False)
    for r, team in enumerate(teams):
        color = TEAM_COLORS[team]
        for c, period in enumerate(periods):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            data = minute_trend(shape[(shape["team"] == team) & (shape["period"] == period)], smooth_min)
            x = data.index + 0.5
            ax.fill_between(x, data["def_line"], data["front_line"], color=color, alpha=0.18, linewidth=0)
            ax.plot(x, data["def_line"], color=color, linewidth=2)
            ax.plot(x, data["centroid"], color=color, linewidth=1, linestyle=(0, (4, 3)))
            ax.plot(x, data["front_line"], color=color, linewidth=1, alpha=0.6)

            # pitch thirds as the only reference lines
            for third in (PITCH_LENGTH / 3, 2 * PITCH_LENGTH / 3):
                ax.axhline(third, color=GRID, linewidth=1, zorder=0)
            for i, (minute, goal_period, scorer, score) in enumerate(scorelines):
                if goal_period != period:
                    continue
                ax.axvline(minute, color=TEXT_MUTED, linewidth=1, linestyle=":")
                # alternate heights so goals minutes apart don't collide
                ax.text(minute + 0.4, PITCH_LENGTH - 3 - 9 * (i % 2),
                        f"{score} {int(minute) + 1}'", fontsize=8, va="top",
                        color=TEXT, fontweight="bold" if scorer == team else "normal")

            start = 45 * (period - 1)
            ax.set_xlim(start, start + half_len[c])
            ax.set_xticks(np.arange(start, start + half_len[c], 15))
            ax.set_xticklabels([f"{m}'" for m in np.arange(start, start + half_len[c], 15).astype(int)])
            ax.set_ylim(0, PITCH_LENGTH)
            ax.set_yticks([0, 35, 70, 105])
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=TEXT_MUTED, labelsize=9, length=0)
            if c == 0:
                ax.set_ylabel("meters from own goal", color=TEXT_MUTED, fontsize=9)
                ax.set_title(f"{team.capitalize()} without the ball", loc="left",
                             color=TEXT, fontsize=11, fontweight="bold")
            else:
                ax.tick_params(left=False)
                ax.spines["left"].set_visible(False)

        # direct labels on the last panel of the row instead of a legend box
        last = axes[r][-1]
        end = minute_trend(shape[(shape["team"] == team) & (shape["period"] == periods[-1])], smooth_min).iloc[-1]
        x_end = 45 * (periods[-1] - 1) + half_len[-1]
        for key, name in (("front_line", "front line"), ("centroid", "centroid"), ("def_line", "defensive line")):
            last.annotate(name, (x_end, end[key]), xytext=(6, 0), textcoords="offset points",
                          fontsize=8, color=TEXT_MUTED, va="center", annotation_clip=False)

    fig.suptitle(title, x=0.125, ha="left", color=TEXT, fontsize=13, fontweight="bold")
    fig.text(0.125, 0.925,
             f"Outfield players only, open play (set pieces and dead ball removed). "
             f"Per-minute medians, {smooth_min}-min rolling mean. Gray lines: pitch thirds. "
             f"Dotted: goals with score (home-away), bold where that team scored.",
             color=TEXT_MUTED, fontsize=9)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
