import numpy as np
import pandas as pd
import pytest

from soccervision.shape import add_depth, average_positions, frame_shape, label_phases

# Home keeper stands near x=0 so home attacks toward x=105; away is the
# mirror image. Four outfield players each (min_players lowered to fit).
HOME = [("H_gk", True, 2.0, 34.0), ("H1", False, 30.0, 10.0), ("H2", False, 30.0, 50.0),
        ("H3", False, 50.0, 10.0), ("H4", False, 50.0, 50.0)]
AWAY = [("A_gk", True, 103.0, 34.0), ("A1", False, 80.0, 20.0), ("A2", False, 80.0, 40.0),
        ("A3", False, 60.0, 20.0), ("A4", False, 60.0, 40.0)]


def _players(frames=(1,)):
    rows = []
    for frame in frames:
        for team, squad in (("home", HOME), ("away", AWAY)):
            for player, is_gk, x, y in squad:
                rows.append(dict(period=1, frame=frame, time_s=frame / 25, team=team,
                                 player=player, is_gk=is_gk, x=x, y=y))
    return pd.DataFrame(rows)


def test_depth_is_distance_from_own_goal():
    p = add_depth(_players())
    home = p[(p.team == "home") & ~p.is_gk].set_index("player").depth
    away = p[(p.team == "away") & ~p.is_gk].set_index("player").depth
    assert home["H1"] == pytest.approx(30.0)
    # away defends x=105, so a player at x=80 is 25 m from their own goal
    assert away["A1"] == pytest.approx(25.0)
    # lateral turns with the team: away (defending x=105) sees y=20 as
    # 48 m from its own left touchline
    lateral = p.set_index("player").lateral
    assert lateral["H1"] == pytest.approx(10.0)
    assert lateral["A1"] == pytest.approx(48.0)


def test_frame_shape_metrics():
    s = frame_shape(add_depth(_players()), min_players=4).set_index("team")
    home = s.loc["home"]
    assert home.n_outfield == 4
    assert home.def_line == pytest.approx(30.0)
    assert home.front_line == pytest.approx(50.0)
    assert home.length == pytest.approx(20.0)
    assert home.width == pytest.approx(40.0)
    assert home.area == pytest.approx(20.0 * 40.0)
    away = s.loc["away"]
    assert away.def_line == pytest.approx(25.0)
    assert away.front_line == pytest.approx(45.0)
    assert away.width == pytest.approx(20.0)


def test_too_few_players_gives_nan_not_a_biased_number():
    s = frame_shape(add_depth(_players()), min_players=8)
    assert s.def_line.isna().all()
    assert (s.n_outfield == 4).all()


def test_label_phases():
    shape = pd.DataFrame(dict(frame=[1, 1, 2, 2, 3, 3], team=["home", "away"] * 3))
    frames = pd.DataFrame(dict(frame=[1, 2, 3],
                               possession=["home", "away", None],
                               in_play=[True, True, False],
                               set_piece=[False, True, False]))
    out = label_phases(shape, frames).set_index(["frame", "team"]).phase
    assert out[(1, "home")] == "in_possession"
    assert out[(1, "away")] == "out_of_possession"
    assert out[(2, "home")] == "set_piece"
    assert out[(3, "away")] == "dead_ball"


def test_average_positions_drop_part_time_players():
    p = add_depth(_players(frames=range(1, 11)))
    # a substitute who only appears in the last frame
    sub = p[p.player == "H1"].tail(1).assign(player="H_sub")
    p = pd.concat([p, sub], ignore_index=True).assign(phase="out_of_possession", window="0-15'")
    avg = average_positions(p, min_share=0.3).set_index("player")
    assert "H_sub" not in avg.index
    assert avg.loc["H3", "depth"] == pytest.approx(50.0)
