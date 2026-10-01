"""Loader for Metrica Sports' open sample tracking data.

Metrica published three anonymized full matches with tracking and event
data (https://github.com/metrica-sports/sample-data). Games 1 and 2 are
CSV: one tracking file per team at 25 fps with positions normalized to
0-1, plus one event file. scripts/download_metrica.py fetches them.

Everything is converted into the source-agnostic layout that shape.py
works on, so the same metrics run on any tracking source (this loader,
the video pipeline, or a provider feed loaded through kloppy):

players -- one row per player per frame:
    period, frame, time_s, team, player, is_gk, x, y
frames -- one row per frame:
    period, frame, time_s, ball_x, ball_y, possession, in_play, set_piece

Coordinates are meters in the pitch.py system: x along the length
(0-105), y across the width (0-68) from the top touchline. Metrica's
normalized y also runs top to bottom, so conversion is a plain scale.
Metrica does not publish the real pitch size of the anonymized games;
105x68 is assumed, which is within ~1 m of any pitch used at that level.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from soccervision.pitch import PITCH_LENGTH, PITCH_WIDTH

FPS = 25

# Events after which the acting team has the ball. CHALLENGE is left out
# on purpose: both teams log one for the same duel, so it says nothing
# about who ends up with the ball -- the RECOVERY that follows does.
POSSESSION_EVENTS = {"PASS", "SET PIECE", "SHOT", "RECOVERY", "BALL LOST"}

# Restarts after which the defending shape is a set-piece setup (walls,
# zonal marking in the box) rather than open-play shape.
SET_PIECE_SUBTYPES = {"CORNER KICK", "FREE KICK"}


def game_dir(root, game):
    return Path(root) / f"Sample_Game_{game}"


def read_team_tracking(path, team):
    """Parse one Metrica team tracking CSV.

    The file has three header rows (team, jersey number, column name);
    only the third is useful. Each player owns two columns, "PlayerN"
    for x and an unnamed one for y. Returns (players, ball) where ball
    has one row per frame.
    """
    raw = pd.read_csv(path, skiprows=2)
    names = list(raw.columns)
    rename = {}
    for i, name in enumerate(names):
        if name.startswith("Player") or name == "Ball":
            rename[name] = f"{name}_x"
            rename[names[i + 1]] = f"{name}_y"
    raw = raw.rename(columns=rename).rename(
        columns={"Period": "period", "Frame": "frame", "Time [s]": "time_s"})

    ball = pd.DataFrame({
        "frame": raw["frame"],
        "ball_x": raw["Ball_x"] * PITCH_LENGTH,
        "ball_y": raw["Ball_y"] * PITCH_WIDTH,
    })

    players = []
    for name in (n for n in names if n.startswith("Player")):
        # substitutes (and sent-off players) are NaN while off the pitch
        on_pitch = raw[f"{name}_x"].notna()
        players.append(pd.DataFrame({
            "period": raw.loc[on_pitch, "period"],
            "frame": raw.loc[on_pitch, "frame"],
            "time_s": raw.loc[on_pitch, "time_s"],
            "team": team,
            "player": name,
            "x": raw.loc[on_pitch, f"{name}_x"] * PITCH_LENGTH,
            "y": raw.loc[on_pitch, f"{name}_y"] * PITCH_WIDTH,
        }))
    players = pd.concat(players, ignore_index=True)

    # Metrica does not label positions. The keeper is the player who is
    # on average furthest from the halfway line -- nobody else spends a
    # whole match within a few meters of a goal line. Distance, not
    # average x: teams swap ends at half-time, so a keeper's average x
    # over the match lands near halfway.
    from_halfway = (players["x"] - PITCH_LENGTH / 2).abs()
    keeper = from_halfway.groupby(players["player"]).mean().idxmax()
    players["is_gk"] = players["player"] == keeper
    return players, ball


def possession_from_events(events, frames, fps=FPS, set_piece_window_s=10.0):
    """Per-frame possession and ball state derived from the event log.

    The event log is a list of change points; each frame takes the state
    set by the most recent one:
    - a POSSESSION_EVENTS event gives the ball to its team (a BALL LOST
      keeps it with the losing team until the opponent's RECOVERY, which
      is when the ball actually changes hands)
    - BALL OUT, a SHOT that ends in a goal or out of play, and a FAULT
      RECEIVED kill the ball until the next event (the restart)
    - after the last event of each period the ball is dead
    Corners and free kicks additionally flag set_piece for the next
    set_piece_window_s seconds.

    frames needs a "frame" column in the tracking file's numbering.
    Returns a copy with possession ("home"/"away"/None), in_play and
    set_piece columns added.
    """
    points = []  # (frame, is_live, team)
    rows = zip(events["Team"].str.lower(), events["Type"], events["Subtype"].fillna(""),
               events["Start Frame"].astype(int), events["End Frame"].astype(int))
    for team, type_, subtype, start, end in rows:
        if type_ in POSSESSION_EVENTS:
            points.append((start, True, team))
        if type_ == "BALL OUT":
            points.append((end, False, None))
        elif type_ == "SHOT" and ("GOAL" in subtype or "OUT" in subtype):
            points.append((end, False, None))
        elif type_ == "FAULT RECEIVED":
            points.append((start, False, None))
    for _, period_events in events.groupby("Period"):
        last_end = int(max(period_events["Start Frame"].max(), period_events["End Frame"].max()))
        points.append((last_end + 1, False, None))

    # at equal frames the dead-ball marker goes first so a restart on the
    # same frame wins
    points.sort(key=lambda p: (p[0], p[1]))
    point_frames = np.array([p[0] for p in points])
    idx = np.searchsorted(point_frames, frames["frame"].to_numpy(), side="right") - 1

    out = frames.copy()
    known = idx >= 0
    live = np.array([p[1] for p in points])
    teams = np.array([p[2] for p in points], dtype=object)
    out["in_play"] = np.where(known, live[np.clip(idx, 0, None)], False)
    out["possession"] = np.where(out["in_play"], teams[np.clip(idx, 0, None)], None)

    window = int(round(set_piece_window_s * fps))
    set_piece = np.zeros(len(out), dtype=bool)
    frame_values = out["frame"].to_numpy()
    restarts = events[(events["Type"] == "SET PIECE") & events["Subtype"].isin(SET_PIECE_SUBTYPES)]
    for start in restarts["Start Frame"]:
        set_piece |= (frame_values >= start) & (frame_values < start + window)
    out["set_piece"] = set_piece
    return out


def load_game(root, game, every_nth=1):
    """Load a Metrica CSV game into the (players, frames) layout.

    every_nth subsamples frames: 5 turns 25 fps into 5 fps, which is
    plenty for shape metrics (nobody moves more than ~2 m in 0.2 s) and
    cuts work five-fold.
    """
    folder = game_dir(root, game)
    prefix = f"Sample_Game_{game}_"
    home, ball = read_team_tracking(folder / f"{prefix}RawTrackingData_Home_Team.csv", "home")
    away, _ = read_team_tracking(folder / f"{prefix}RawTrackingData_Away_Team.csv", "away")
    events = pd.read_csv(folder / f"{prefix}RawEventsData.csv")

    players = pd.concat([home, away], ignore_index=True)
    frames = (players[["period", "frame", "time_s"]].drop_duplicates("frame")
              .merge(ball, on="frame").sort_values("frame").reset_index(drop=True))
    frames = possession_from_events(events, frames)

    if every_nth > 1:
        keep = frames["frame"][(frames["frame"] - 1) % every_nth == 0]
        frames = frames[frames["frame"].isin(keep)].reset_index(drop=True)
        players = players[players["frame"].isin(keep)].reset_index(drop=True)
    return players, frames, events
