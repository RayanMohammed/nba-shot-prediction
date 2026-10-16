import pandas as pd

from nba_shots.features import (add_angle, add_distance_hinges, add_game_clock, add_is_home,
                                 common_action_types)


def test_hinges_count_feet_past_each_knot():
    df = add_distance_hinges(pd.DataFrame({"shot_distance": [0, 2, 5, 30]}), knots=[1, 3, 23])
    assert list(df["dist_over_1"]) == [0, 1, 4, 29]
    assert list(df["dist_over_3"]) == [0, 0, 2, 27]
    assert list(df["dist_over_23"]) == [0, 0, 0, 7]



def test_angle_straight_on_baseline_and_both_sides_equal():
    df = add_angle(pd.DataFrame({"loc_x": [0.0, 10.0, -10.0], "loc_y": [20.0, 5.25, 5.25]}))
    assert list(df["angle"].round(1)) == [0.0, 90.0, 90.0]


def test_game_clock():
    df = add_game_clock(pd.DataFrame({"mins_left": [11, 0], "secs_left": [30, 4], "quarter": [1, 6]}))
    assert list(df["secs_left_in_period"]) == [690, 4]
    assert list(df["last_5_secs"]) == [0, 1]
    assert list(df["period"]) == [1, 5]  # double overtime grouped with all overtimes


def test_is_home_finds_each_teams_abbreviation():
    # Team A (AAA) hosts BBB in g1, then plays at CCC in g2.
    # Team B (BBB) plays at AAA in g1, then hosts DDD in g3.
    df = pd.DataFrame({
        "season": "2018-19",
        "team_name": ["Team A", "Team A", "Team B", "Team B"],
        "game_key": ["g1", "g2", "g1", "g3"],
        "home_team": ["AAA", "CCC", "AAA", "BBB"],
        "away_team": ["BBB", "AAA", "BBB", "DDD"],
    })
    assert list(add_is_home(df)["is_home"]) == [1, 0, 0, 1]


def test_rare_action_types_left_out():
    train = pd.DataFrame({"action_type": ["Jump Shot"] * 5 + ["Dunk"] * 2 + ["Hook"]})
    assert common_action_types(train, min_shots=2) == ["Jump Shot", "Dunk"]
