import numpy as np
import pandas as pd

from soccervision.metrica import possession_from_events, read_team_tracking

TRACKING_CSV = """\
,,,Home,,Home,,Home,,,
,,,11,,1,,2,,,
Period,Frame,Time [s],Player11,,Player1,,Player2,,Ball,
1,1,0.04,0.02,0.50,0.30,0.40,NaN,NaN,0.50,0.50
1,2,0.08,0.02,0.50,0.31,0.40,NaN,NaN,0.51,0.50
1,3,0.12,0.02,0.50,0.32,0.40,0.60,0.25,0.52,0.50
2,4,0.16,0.98,0.50,0.70,0.40,0.40,0.25,0.52,0.50
2,5,0.20,0.98,0.50,0.70,0.40,0.40,0.25,0.52,0.50
2,6,0.24,0.98,0.50,0.70,0.40,0.40,0.25,0.52,0.50
"""


def test_read_team_tracking(tmp_path):
    path = tmp_path / "home.csv"
    path.write_text(TRACKING_CSV)
    players, ball = read_team_tracking(path, "home")

    # Player2 is a substitute: absent (NaN) until frame 3, so one row only
    assert players.groupby("player").size().to_dict() == {"Player1": 6, "Player11": 6, "Player2": 4}
    # normalized 0-1 coordinates become meters on a 105x68 pitch
    p1 = players[(players.player == "Player1") & (players.frame == 1)].iloc[0]
    assert np.isclose(p1.x, 0.30 * 105) and np.isclose(p1.y, 0.40 * 68)
    # keeper is the player standing closest to a goal line -- even though
    # swapping ends at half-time puts their whole-match average at halfway
    assert players[players.is_gk].player.unique().tolist() == ["Player11"]
    assert np.isclose(ball.ball_x.iloc[0], 0.5 * 105)


def _event(team, type_, start, end=None, subtype=None, period=1):
    return {"Team": team, "Type": type_, "Subtype": subtype, "Period": period,
            "Start Frame": start, "End Frame": start if end is None else end}


def test_possession_from_events():
    events = pd.DataFrame([
        _event("Home", "SET PIECE", 1, subtype="KICK OFF"),
        _event("Home", "PASS", 1, 5),
        _event("Home", "BALL LOST", 10, 14, subtype="INTERCEPTION"),
        _event("Away", "RECOVERY", 15),
        _event("Away", "BALL OUT", 18, 20),
        _event("Home", "SET PIECE", 30, subtype="THROW IN"),
        _event("Away", "FAULT RECEIVED", 40),
        _event("Away", "SET PIECE", 45, subtype="FREE KICK"),
        _event("Away", "SHOT", 50, 55, subtype="ON TARGET-GOAL"),
        _event("Home", "SET PIECE", 70, subtype="KICK OFF"),
        _event("Home", "BALL LOST", 76, 77, subtype="END HALF"),
    ])
    frames = pd.DataFrame({"frame": np.arange(1, 81), "period": 1})
    out = possession_from_events(events, frames, fps=5, set_piece_window_s=2).set_index("frame")

    assert out.loc[3, "possession"] == "home" and out.loc[3, "in_play"]
    # still Home's until the other team actually recovers the ball
    assert out.loc[12, "possession"] == "home"
    assert out.loc[16, "possession"] == "away"
    # ball out of play from the moment it crosses the line until the restart
    assert not out.loc[20, "in_play"] and not out.loc[25, "in_play"]
    assert out.loc[30, "in_play"] and out.loc[30, "possession"] == "home"
    # a foul stops play until the free kick
    assert not out.loc[42, "in_play"]
    assert out.loc[45, "possession"] == "away"
    # free kicks open a set-piece window (2 s at 5 fps = 10 frames); throw-ins do not
    assert out.loc[47, "set_piece"] and not out.loc[56, "set_piece"]
    assert not out.loc[31, "set_piece"]
    # a goal kills the ball until the kick off
    assert not out.loc[60, "in_play"]
    assert out.loc[75, "possession"] == "home"
    # after the last event of the period the ball is dead
    assert not out.loc[80, "in_play"]
