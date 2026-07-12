"""Formation detection from top-down player positions.

For each possession phase, each team's outfield players get an average
pitch position. Those positions are normalized into the team's own
bounding box (so compactness and pitch location don't matter, only
shape) and matched against reference formation templates with the
Hungarian algorithm. The template with the lowest mean matched distance
wins.

Templates are laid out in a normalized space where x runs from the
team's own goal (0) toward the opponent goal (1) and y across the
width. Since the true attack direction is not known from tracking data
alone, matching is tried under horizontal and vertical flips and the
best orientation is kept.

Honest limitations:
- Needs (close to) all 10 outfield players in view with valid pitch
  coordinates. Broadcast cameras rarely show all 22 players, so on
  typical footage many phases can't be classified at all. The result
  includes how many players were actually used -- treat anything
  below 10 with suspicion.
- Distinguishing e.g. 4-4-2 from 4-2-3-1 from average positions is
  ambiguous even for humans; the score margins are reported so close
  calls are visible.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def _rows(*counts):
    """Build a template from lines: e.g. _rows(4, 4, 2) -> 4-4-2.
    Defenders at low x, attackers at high x, each line spread in y."""
    positions = []
    n_rows = len(counts)
    for r, count in enumerate(counts):
        x = (r + 1) / (n_rows + 1)
        for i in range(count):
            y = (i + 1) / (count + 1)
            positions.append((x, y))
    return np.array(positions)


FORMATION_TEMPLATES = {
    "4-4-2": _rows(4, 4, 2),
    "4-3-3": _rows(4, 3, 3),
    "4-2-3-1": _rows(4, 2, 3, 1),
    "4-1-4-1": _rows(4, 1, 4, 1),
    "4-5-1": _rows(4, 5, 1),
    "3-5-2": _rows(3, 5, 2),
    "3-4-3": _rows(3, 4, 3),
    "5-3-2": _rows(5, 3, 2),
    "5-4-1": _rows(5, 4, 1),
}


def _normalize(positions):
    """Scale positions into [0,1]^2 using the team's own extent."""
    positions = np.asarray(positions, dtype=float)
    span = positions.max(axis=0) - positions.min(axis=0)
    span[span < 1e-6] = 1.0
    return (positions - positions.min(axis=0)) / span


def _match_cost(observed, template):
    """Mean distance after optimal player-to-role assignment. Works for
    fewer than 10 observed players (rectangular assignment), at the cost
    of the template roles left unmatched -- hence the player count is
    reported alongside the result."""
    diff = observed[:, None, :] - template[None, :, :]
    cost = np.linalg.norm(diff, axis=2)
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].mean())


def classify_shape(avg_positions):
    """avg_positions: (N, 2) mean pitch coords of one team's outfield
    players. Returns ranked list of (formation, cost)."""
    observed = _normalize(avg_positions)
    results = []
    for name, template in FORMATION_TEMPLATES.items():
        best = np.inf
        for flip_x in (False, True):
            for flip_y in (False, True):
                t = template.copy()
                if flip_x:
                    t[:, 0] = 1 - t[:, 0]
                if flip_y:
                    t[:, 1] = 1 - t[:, 1]
                best = min(best, _match_cost(observed, _normalize(t)))
        results.append((name, best))
    results.sort(key=lambda r: r[1])
    return results


def phase_average_positions(tracks, start, end, team, min_observations=None):
    """Average pitch position per outfield track id over a frame range.
    Only counts frames where the player had a valid pitch coordinate.
    Returns (track_ids, positions)."""
    if min_observations is None:
        min_observations = max((end - start) // 4, 5)

    sums, counts = {}, {}
    for i in range(start, end):
        for tid, info in tracks["players"][i].items():
            if info.get("team") != team or info.get("cls") == "goalkeeper":
                continue
            pos = info.get("position_pitch")
            if pos is None:
                continue
            sums[tid] = sums.get(tid, np.zeros(2)) + np.asarray(pos)
            counts[tid] = counts.get(tid, 0) + 1

    ids = [t for t, c in counts.items() if c >= min_observations]
    # ID switches can leave more than 10 qualifying tracks; keep the ten
    # with the most observations
    ids.sort(key=lambda t: counts[t], reverse=True)
    ids = ids[:10]
    positions = np.array([sums[t] / counts[t] for t in ids]) if ids else np.zeros((0, 2))
    return ids, positions


def analyze_phases(tracks, segments, min_players=7):
    """Formation classification per possession segment, per team.

    Each segment yields one entry per team: the team in possession
    (attacking shape) and the team out of possession (defensive shape).
    Segments where a team has fewer than min_players usable outfield
    positions are marked unclassified rather than force-fit.
    """
    report = []
    for seg in segments:
        for team in (1, 2):
            ids, positions = phase_average_positions(tracks, seg["start"], seg["end"], team)
            entry = {
                "segment": {"start": seg["start"], "end": seg["end"]},
                "team": team,
                "phase": "in_possession" if team == seg["team"] else "out_of_possession",
                "players_used": len(ids),
                "track_ids": ids,
                "avg_positions": positions.tolist(),
            }
            if len(ids) >= min_players:
                ranking = classify_shape(positions)
                entry["formation"] = ranking[0][0]
                entry["match_cost"] = round(ranking[0][1], 4)
                entry["ranking"] = [(n, round(c, 4)) for n, c in ranking[:3]]
                entry["reliable"] = len(ids) == 10
            else:
                entry["formation"] = None
                entry["reason"] = f"only {len(ids)} outfield players with pitch coords"
            report.append(entry)
    return report
