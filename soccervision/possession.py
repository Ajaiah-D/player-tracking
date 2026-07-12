"""Ball possession estimation.

Per frame, the ball is assigned to the nearest player foot point within
a pixel threshold (same heuristic as the reference player_ball_assigner).
The raw per-frame team signal is noisy -- the ball is often nearest to
nobody, or briefly nearest to an opponent during a pass -- so it is
carried forward through unassigned frames and median-smoothed before
being segmented into possession phases.

This is a heuristic, not ground truth. Pixel distance ignores
perspective (a fixed threshold means different real distances at the
top and bottom of the frame) and interpolated ball positions during
detection gaps can attach the ball to the wrong player.
"""

import numpy as np


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def nearest_player(players, ball_position, max_distance=70.0):
    best_id, best_d = None, max_distance
    for tid, info in players.items():
        b = info["bbox"]
        d = min(
            _dist((b[0], b[3]), ball_position),
            _dist((b[2], b[3]), ball_position),
        )
        if d < best_d:
            best_id, best_d = tid, d
    return best_id


def possession_by_frame(tracks, max_distance=70.0, smooth_window=25):
    """Returns (team_per_frame, holder_per_frame). Team is 1, 2 or 0
    when nothing sensible can be assigned (e.g. before first touch)."""
    n = len(tracks["players"])
    raw = np.zeros(n, dtype=int)
    holders = [None] * n

    for i in range(n):
        ball = tracks["ball"][i].get(1)
        if ball is None:
            raw[i] = raw[i - 1] if i > 0 else 0
            continue
        tid = nearest_player(tracks["players"][i], ball["position"], max_distance)
        if tid is not None:
            holders[i] = tid
            team = tracks["players"][i][tid].get("team", 0)
            raw[i] = team if team in (1, 2) else (raw[i - 1] if i > 0 else 0)
        else:
            raw[i] = raw[i - 1] if i > 0 else 0

    # median filter to suppress single-frame flips
    smoothed = raw.copy()
    half = smooth_window // 2
    for i in range(n):
        window = raw[max(0, i - half) : i + half + 1]
        window = window[window != 0]
        if len(window):
            smoothed[i] = int(np.median(window))
    return smoothed, holders


def possession_segments(team_per_frame, min_frames=25):
    """Contiguous runs of one team holding the ball, shorter runs are
    merged into their neighbours by the smoothing above or dropped."""
    segments = []
    start = 0
    for i in range(1, len(team_per_frame) + 1):
        if i == len(team_per_frame) or team_per_frame[i] != team_per_frame[start]:
            team = int(team_per_frame[start])
            if team != 0 and (i - start) >= min_frames:
                segments.append({"team": team, "start": start, "end": i})
            start = i
    return segments
