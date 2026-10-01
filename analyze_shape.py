"""Team shape analysis on full-match tracking data.

Loads a Metrica sample game, computes per-frame shape metrics for both
teams (defensive line, front line, centroid, length, width, area),
splits them by phase of play from the event data, and writes:

    output/shape/metrica_game<n>/shape_by_frame.csv   every sampled frame
    output/shape/metrica_game<n>/summary.csv          medians per team/phase/15 min
    output/shape/metrica_game<n>/block_height.png     out-of-possession block over time
    output/shape/metrica_game<n>/report.html          match report for staff (open in a browser)

    python scripts/download_metrica.py --game 1
    python analyze_shape.py --game 1 --open      # --open shows the report when done
"""

import argparse
import webbrowser
from pathlib import Path

import pandas as pd

from soccervision.metrica import load_game
from soccervision.report import build_report_data, write_report
from soccervision.shape import add_depth, frame_shape, label_phases, summarize
from soccervision.shape_plot import plot_block_height


def goal_minutes(events, frames):
    """(minute, period, scoring_team) for every goal in the event log.

    Goals are the events whose subtype ends in "-GOAL" -- not only shots:
    game 1 has one logged as BALL OUT / WOODWORK-GOAL. The scorer is read
    from the restart, not the event's team: the team that kicks off next
    conceded, which also gets own goals right.
    """
    period_start = frames.groupby("period")["time_s"].min()
    subtype = events["Subtype"].fillna("")
    goals = events[subtype.str.endswith("-GOAL")]
    kick_offs = events[subtype == "KICK OFF"]
    out = []
    for idx, row in goals.iterrows():
        restart = kick_offs[(kick_offs.index > idx) & (kick_offs["Period"] == row["Period"])].head(1)
        if len(restart):
            scorer = "home" if restart["Team"].iloc[0] == "Away" else "away"
        else:  # goal right before half-time or the final whistle
            scorer = row["Team"].lower()
        minute = (row["Start Time [s]"] - period_start[row["Period"]]) / 60 + 45 * (row["Period"] - 1)
        out.append((minute, row["Period"], scorer))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", type=int, choices=(1, 2), default=1)
    parser.add_argument("--data-root", type=Path, default=Path("data/metrica"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/shape"))
    parser.add_argument("--every-nth", type=int, default=5,
                        help="frame subsampling; 5 turns Metrica's 25 fps into 5 fps")
    parser.add_argument("--open", action="store_true", help="open the match report in the default browser")
    args = parser.parse_args()

    out_dir = args.output_dir / f"metrica_game{args.game}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading Metrica sample game {args.game}")
    players, frames, events = load_game(args.data_root, args.game, every_nth=args.every_nth)
    print(f"{frames['frame'].nunique()} frames, {players['player'].nunique()} players, {len(events)} events")
    in_play = frames["in_play"].mean()
    print(f"ball in play {in_play:.0%} of frames; possession split "
          + ", ".join(f"{t} {s:.0%}" for t, s in frames["possession"].value_counts(normalize=True).items()))

    print("computing shape metrics")
    players = add_depth(players)
    shape = label_phases(frame_shape(players), frames)
    shape.to_csv(out_dir / "shape_by_frame.csv", index=False)

    summary = summarize(shape)
    summary.to_csv(out_dir / "summary.csv", index=False)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(summary.to_string(index=False))

    goals = goal_minutes(events, frames)
    print("goals: " + ", ".join(f"{team} {int(m)}'" for m, _, team in goals))
    plot_block_height(shape, out_dir / "block_height.png", goals=goals,
                      title=f"Metrica sample game {args.game}: out-of-possession block")

    print("building match report")
    report = build_report_data(players, frames, shape, summary, goals,
                               page_title=f"Game {args.game} Shape Report",
                               source=f"Metrica Sports sample game {args.game}")
    report_path = out_dir / "report.html"
    write_report(report, report_path)
    print(f"outputs -> {out_dir}")
    print(f"match report -> {report_path.resolve()}")
    if args.open:
        webbrowser.open(report_path.resolve().as_uri())


if __name__ == "__main__":
    main()
