"""Team shape metrics from top-down tracking data.

Works on the source-agnostic layout described in metrica.py: a long
players table (period, frame, time_s, team, player, is_gk, x, y in
pitch.py meters) and a per-frame table with possession, in_play and
set_piece. Any tracking source converted to that layout gets the same
metrics.

Depth is measured from the team's OWN goal line, so both teams read the
same way -- "defensive line 38 m" means 38 m from the goal it defends --
no matter which end they attack or that they swap ends at half-time.

Per frame and team, over outfield players only (the keeper would drag
every depth metric toward goal and says nothing about the block):
- def_line: depth of the deepest outfield player, i.e. the offside line
- front_line: depth of the most advanced outfield player
- centroid: mean depth of the outfield players
- length: front_line - def_line (vertical compactness)
- width: y-spread of the outfield players (horizontal compactness)
- area: convex hull area of the outfield players (overall compactness)
"""

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

from soccervision.pitch import PITCH_LENGTH, PITCH_WIDTH

METRICS = ["def_line", "front_line", "centroid", "length", "width", "area"]


def add_depth(players):
    """Add "depth" (distance from the team's own goal line) and "lateral"
    (y rotated with the team, so 0 is always the team's left touchline
    when facing the opponent's goal) columns.

    Which goal a team defends is read per period: the team whose keeper
    stands further left defends the left goal. If a keeper is not tagged
    (broadcast tracking often misses them) the team's average position
    is used instead -- over a half, each team's average still sits on
    its own side of the opponent's.
    """
    players = players.copy()
    side_x = players.groupby(["period", "team"])["x"].mean()
    keepers = players[players["is_gk"]].groupby(["period", "team"])["x"].mean()
    side_x.loc[keepers.index] = keepers

    # the other team's value in the same period (two teams per period)
    other = side_x.groupby("period").transform("sum") - side_x
    defends_left = (side_x < other).rename("defends_left")

    players = players.join(defends_left, on=["period", "team"])
    players["depth"] = np.where(players["defends_left"], players["x"], PITCH_LENGTH - players["x"])
    # switching ends is a 180-degree turn, so the touchlines swap too
    players["lateral"] = np.where(players["defends_left"], players["y"], PITCH_WIDTH - players["y"])
    return players.drop(columns="defends_left")


def match_minute(df):
    """Match clock in minutes: 0-45+ in the first half, 45-90+ in the second."""
    period_start = df.groupby("period")["time_s"].transform("min")
    return (df["time_s"] - period_start) / 60 + 45 * (df["period"] - 1)


def window_labels(minute, period, window_min=15):
    """Label like "15-30'" per row; stoppage time folds into the last
    window of each half."""
    half_minute = minute - 45 * (period - 1)
    last_window = 45 // window_min - 1
    window_idx = np.minimum(half_minute // window_min, last_window).astype(int)
    start = window_idx * window_min + 45 * (period - 1)
    return np.array([f"{s}-{s + window_min}'" for s in start])


def _hull_area(points):
    if len(points) < 3:
        return 0.0
    try:
        return float(ConvexHull(points).volume)  # "volume" is area in 2-D
    except QhullError:  # all players on one line
        return 0.0


def frame_shape(players, min_players=8):
    """Shape metrics per (frame, team). players must have a depth column.

    Frames where fewer than min_players outfield players are visible get
    NaN metrics: with players missing, width/length/area are silently
    underestimated, and a missing number is better than a wrong one.
    Full-pitch tracking (Metrica, providers) almost always has 10; for
    broadcast-derived tracking this filter matters.
    """
    outfield = players[~players["is_gk"]]
    grouped = outfield.groupby(["period", "frame", "team"], sort=True)
    shape = grouped.agg(
        time_s=("time_s", "first"),
        n_outfield=("depth", "size"),
        def_line=("depth", "min"),
        front_line=("depth", "max"),
        centroid=("depth", "mean"),
        y_min=("y", "min"),
        y_max=("y", "max"),
    )
    shape["length"] = shape["front_line"] - shape["def_line"]
    shape["width"] = shape["y_max"] - shape["y_min"]

    xy = outfield[["depth", "y"]].to_numpy()
    areas = {key: _hull_area(xy[idx]) for key, idx in grouped.indices.items()}
    shape["area"] = pd.Series(areas)

    shape.loc[shape["n_outfield"] < min_players, METRICS] = np.nan
    shape = shape.drop(columns=["y_min", "y_max"]).reset_index()
    shape["minute"] = match_minute(shape)
    return shape


def label_phases(shape, frames):
    """Tag each (frame, team) row with the team's phase of play:
    in_possession, out_of_possession, set_piece or dead_ball."""
    merged = shape.merge(frames[["frame", "possession", "in_play", "set_piece"]], on="frame", how="left")
    phase = np.where(merged["possession"] == merged["team"], "in_possession", "out_of_possession")
    phase = np.where(merged["set_piece"].fillna(False).astype(bool), "set_piece", phase)
    phase = np.where(merged["in_play"].fillna(False).astype(bool), phase, "dead_ball")
    merged["phase"] = phase
    return merged.drop(columns=["possession", "in_play", "set_piece"])


def summarize(shape, window_min=15, phases=("in_possession", "out_of_possession")):
    """Median shape per team, phase and time window.

    Medians rather than means: a single breakaway or a defender stepping
    out to press shifts a mean noticeably, but not the typical shape.
    Stoppage time is folded into the last window of each half.
    """
    shape = shape[shape["phase"].isin(phases)].copy()
    shape["window"] = window_labels(shape["minute"], shape["period"], window_min)

    dt = float(np.median(np.diff(np.sort(shape["time_s"].unique()))))
    rows = shape.groupby(["team", "phase", "period", "window"])
    summary = rows[METRICS].median().round(1)
    summary.insert(0, "minutes", (rows.size() * dt / 60).round(1))
    return summary.reset_index()


def minute_trend(team_shape, smooth_min=5, metrics=("def_line", "centroid", "front_line", "width", "length")):
    """Per-minute medians of one team's rows, then a centered rolling
    mean so a line shows the trend rather than every possession. Minutes
    with under 10 s of data are left out of the average. Index is the
    whole match minute."""
    team_shape = team_shape.assign(bin=team_shape["minute"].astype(int))
    grouped = team_shape.groupby("bin")
    dt = float(np.median(np.diff(np.sort(team_shape["time_s"].unique()))))
    per_min = grouped[list(metrics)].median()
    per_min = per_min[grouped.size() * dt >= 10]
    return per_min.rolling(smooth_min, center=True, min_periods=1).mean()


def average_positions(players, min_share=0.3):
    """Mean (depth, lateral) per player for each team, phase and window
    column present in players (e.g. "window"). Players on the pitch for
    less than min_share of their team's frames in that slice (late subs,
    the player they replaced) are left out so the picture shows the
    shape that was actually played. Outfield and keeper both included."""
    keys = ["team", "phase", "window"]
    per_player = players.groupby(keys + ["player", "is_gk"]).agg(
        depth=("depth", "mean"), lateral=("lateral", "mean"), frames=("frame", "size")).reset_index()
    team_frames = players.groupby(keys)["frame"].nunique().rename("team_frames")
    per_player = per_player.join(team_frames, on=keys)
    per_player = per_player[per_player["frames"] >= min_share * per_player["team_frames"]]
    # over a long window a sub and the player replaced can both clear the
    # share; keep the ten outfield players who were on longest
    per_player = per_player.sort_values("frames", ascending=False)
    rank = per_player.groupby(keys + ["is_gk"]).cumcount()
    per_player = per_player[per_player["is_gk"] | (rank < 10)]
    return per_player.drop(columns=["frames", "team_frames"]).sort_values(keys + ["player"])
