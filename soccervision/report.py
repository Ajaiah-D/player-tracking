"""Match shape report: one self-contained HTML page per match for
coaching staff (open it in any browser, no Python needed).

The page is report_template.html with the match data embedded as JSON.
Everything the page shows is computed here; the template only draws it.
"""

import html
import json
from pathlib import Path

import numpy as np

from soccervision.pitch import PITCH_LENGTH, PITCH_WIDTH
from soccervision.shape import average_positions, label_phases, match_minute, minute_trend, window_labels

TEMPLATE = Path(__file__).with_name("report_template.html")
PHASES = ("out_of_possession", "in_possession")
TEAMS = ("home", "away")


def _r(value, digits=1):
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def _dash(window):
    """"15-30'" -> "15–30′" for display."""
    return window.replace("-", "–").replace("'", "′")


def _series(shape):
    out = {}
    for team in TEAMS:
        out[team] = {}
        for phase in PHASES:
            rows = []
            for period in sorted(shape["period"].unique()):
                sel = shape[(shape["team"] == team) & (shape["phase"] == phase) & (shape["period"] == period)]
                trend = minute_trend(sel)
                for minute, row in trend.iterrows():
                    rows.append({"m": int(minute), "p": int(period),
                                 **{k: _r(row[k]) for k in trend.columns}})
            out[team][phase] = rows
    return out


def _shapes(players, frames):
    """Average positions per team, phase and window, with depth/lateral
    in the team's own frame (own goal at 0, attacking toward 105)."""
    phased = label_phases(players, frames)
    phased = phased[phased["phase"].isin(PHASES)].copy()
    phased["minute"] = match_minute(phased)
    fifteen = phased.assign(window=window_labels(phased["minute"], phased["period"]))
    halves = phased.assign(window=np.where(phased["period"] == 1, "1st half", "2nd half"))
    full = phased.assign(window="Full match")

    out = {}
    for slice_ in (full, halves, fifteen):
        avg = average_positions(slice_)
        for (team, phase, window), grp in avg.groupby(["team", "phase", "window"]):
            out.setdefault(team, {}).setdefault(phase, {})[window] = [
                {"n": p.replace("Player", ""), "gk": bool(gk), "x": _r(d), "y": _r(l)}
                for p, gk, d, l in zip(grp["player"], grp["is_gk"], grp["depth"], grp["lateral"])]
    return out


def _findings(shape, summary, names):
    """A few plain-language findings per team, computed from the data
    (thresholds keep them from reporting noise as change)."""
    out = {}
    for team in TEAMS:
        name = names[team]
        oop = shape[(shape["team"] == team) & (shape["phase"] == "out_of_possession")]
        ip = shape[(shape["team"] == team) & (shape["phase"] == "in_possession")]
        h1 = oop.loc[oop["period"] == 1, "def_line"].median()
        h2 = oop.loc[oop["period"] == 2, "def_line"].median()
        diff = h2 - h1
        lines = []
        if abs(diff) >= 3:
            direction = "deeper" if diff < 0 else "higher"
            lines.append(f"Defended {abs(diff):.0f} m {direction} after half-time: "
                         f"defensive line {h1:.0f} m → {h2:.0f} m from goal.")
        else:
            lines.append(f"Held a similar defensive line in both halves "
                         f"({h1:.0f} m and {h2:.0f} m from goal).")

        windows = summary[(summary["team"] == team) & (summary["phase"] == "out_of_possession")]
        hi = windows.loc[windows["def_line"].idxmax()]
        lo = windows.loc[windows["def_line"].idxmin()]
        lines.append(f"Highest block {_dash(hi['window'])} ({hi['def_line']:.0f} m), "
                     f"deepest {_dash(lo['window'])} ({lo['def_line']:.0f} m).")

        w_out, w_in = oop["width"].median(), ip["width"].median()
        lines.append(f"{w_out:.0f} m wide without the ball, {w_in:.0f} m with it; "
                     f"{oop['length'].median():.0f} m from back line to front line when defending.")
        out[team] = lines
    return out


def build_report_data(players, frames, shape, summary, goals, page_title, source, names=None):
    """players: add_depth output; shape: label_phases output; summary:
    summarize output; goals: [(minute, period, scoring_team)]."""
    names = names or {"home": "Home", "away": "Away"}
    tally = {"home": 0, "away": 0}
    goal_rows = []
    for minute, period, scorer in sorted(goals, key=lambda g: (g[1], g[0])):
        tally[scorer] += 1
        goal_rows.append({"m": _r(minute, 2), "p": int(period), "team": scorer,
                          "score": f"{tally['home']}–{tally['away']}"})

    halves = []
    for period in sorted(shape["period"].unique()):
        minutes = shape.loc[shape["period"] == period, "minute"]
        halves.append({"p": int(period), "start": 45 * (int(period) - 1), "end": _r(minutes.max(), 2)})

    possession = frames["possession"].value_counts(normalize=True)
    table = summary.assign(label=summary["window"].map(_dash))
    return {
        "page_title": page_title,
        "source": source,
        "names": names,
        "score": [tally["home"], tally["away"]],
        "goals": goal_rows,
        "halves": halves,
        "pitch": [PITCH_LENGTH, PITCH_WIDTH],
        "possession": {t: _r(possession.get(t, 0.0), 3) for t in TEAMS},
        "in_play": _r(frames["in_play"].mean(), 3),
        "findings": _findings(shape, summary, names),
        "series": _series(shape),
        "shapes": _shapes(players, frames),
        "table": json.loads(table.to_json(orient="records")),
    }


def write_report(data, path, standalone=True):
    """Fill the template. standalone adds the <!doctype>/<html> wrapper
    so the file opens correctly straight from disk."""
    page = TEMPLATE.read_text(encoding="utf-8")
    # "</" inside the JSON would end the <script> block early
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = page.replace("__PAGE_TITLE__", html.escape(data["page_title"]))
    page = page.replace("__REPORT_DATA__", payload)
    if standalone:
        page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1">\n</head>\n<body>\n'
                + page + "\n</body>\n</html>\n")
    Path(path).write_text(page, encoding="utf-8")
