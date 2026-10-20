import numpy as np
import pandas as pd

from nba_shots.features import (add_angle, add_defense_history, add_distance_hinges, add_game_clock,
                                 add_is_home, add_opponent, add_opponent_defense, add_shooter_history,
                                 add_shooter_skill, all_features, common_action_types, shot_family)


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


def test_shot_families():
    assert shot_family("Driving Floating Bank Jump Shot") == "floater"
    assert shot_family("Putback Dunk Shot") == "dunk"           # dunk is checked before putback
    assert shot_family("Tip Layup Shot") == "tip/putback"       # tip is checked before layup
    assert shot_family("Cutting Finger Roll Layup Shot") == "layup"
    assert shot_family("Turnaround Bank Hook Shot") == "hook"   # hook is checked before turnaround
    assert shot_family("Step Back Jump shot") == "step-back"
    assert shot_family("Running Pull-Up Jump Shot") == "pull-up"
    assert shot_family("Jump Bank Shot") == "jumper"            # "Bank" alone doesn't make a family
    assert shot_family("Jump Shot") == "jumper"


def test_family_flags():
    df = pd.DataFrame({
        "season": "2018-19", "team_name": "Team A", "game_key": "g1", "home_team": "AAA",
        "away_team": "BBB", "shot_distance": [1, 1, 20], "is_three": 0, "loc_x": 0.0,
        "loc_y": [6.0, 6.0, 25.0], "mins_left": 5, "secs_left": 0, "quarter": 1,
        "basic_zone": "Restricted Area", "zone_name": "Center",
        "action_type": ["Driving Layup Shot", "Alley Oop Dunk Shot", "Jump Shot"],
    })
    X = all_features(df, common_actions=[], shot_type="family+flags")
    assert list(X["on_the_move"]) == [1, 0, 0]
    assert list(X["alley_oop"]) == [0, 1, 0]
    assert list(X["family_layup"]) == [1, 0, 0]


def skill_shots():
    """One player, three games in date order, all midrange twos (category "two"), plus one three."""
    return pd.DataFrame({
        "player_id": "p1",
        "game_key": ["g1", "g1", "g2", "g3", "g3"],
        "game_date": pd.to_datetime(["2019-01-01", "2019-01-01", "2019-01-05", "2019-01-09", "2019-01-09"]),
        "action_type": "Jump Shot",
        "basic_zone": ["Mid-Range", "Mid-Range", "Mid-Range", "Mid-Range", "Above the Break 3"],
        "is_three": [0, 0, 0, 0, 1],
        "made": [1, 1, 0, 1, 1],
    })


def expected_40pct():
    return pd.DataFrame({"family": ["jumper", "jumper"], "basic_zone": ["Mid-Range", "Above the Break 3"],
                         "expected": [0.4, 0.4]})


def test_shooter_history_uses_only_earlier_games():
    out = add_shooter_history(skill_shots(), expected_40pct())
    # game 1: no history. game 2: two makes in g1 -> 2 * (1 - 0.4) = 1.2. game 3 (twos): g1 + g2 -> 1.2 - 0.4 = 0.8
    assert list(out["shooter_prior_shots"]) == [0, 0, 2, 3, 0]   # the three has its own (empty) history
    assert np.allclose(out["shooter_prior_resid"], [0, 0, 1.2, 0.8, 0])


def test_shooter_history_ignores_own_and_later_outcomes():
    before = add_shooter_history(skill_shots(), expected_40pct())
    changed = skill_shots()
    changed.loc[2, "made"] = 1   # flip the outcome of the game-2 shot...
    after = add_shooter_history(changed, expected_40pct())
    # (np.isclose instead of ==: subtracting from a running total can leave a rounding crumb like 1.1999999999999997)
    assert np.isclose(after.loc[2, "shooter_prior_resid"], before.loc[2, "shooter_prior_resid"])  # its own value can't change
    assert np.isclose(after.loc[0, "shooter_prior_resid"], before.loc[0, "shooter_prior_resid"])  # earlier games can't change
    assert not np.isclose(after.loc[3, "shooter_prior_resid"], before.loc[3, "shooter_prior_resid"])  # a later game DOES see it


def test_shooter_skill_shrinks_toward_zero():
    df = pd.DataFrame({"shooter_prior_resid": [5.0, 5.0], "shooter_prior_shots": [10, 1000]})
    out = add_shooter_skill(df, k=90)
    assert np.allclose(out["shooter_skill"], [5 / 100, 5 / 1090])  # few shots -> pulled hard toward 0


def test_opponent_is_the_other_team():
    # Team A (AAA) hosts BBB in g1, then plays at CCC in g2.  Team B (BBB) plays at AAA in g1, hosts DDD in g3.
    df = pd.DataFrame({
        "season": "2018-19",
        "team_name": ["Team A", "Team A", "Team B", "Team B"],
        "game_key": ["g1", "g2", "g1", "g3"],
        "home_team": ["AAA", "CCC", "AAA", "BBB"],
        "away_team": ["BBB", "AAA", "BBB", "DDD"],
    })
    assert list(add_opponent(df)["opponent"]) == ["BBB", "CCC", "AAA", "DDD"]


def test_defense_history_restarts_each_season():
    # Team A shoots against BBB in three games: two in 2017-18, one in 2018-19
    df = pd.DataFrame({
        "season": ["2017-18", "2017-18", "2018-19"],
        "team_name": "Team A",
        "game_key": ["g1", "g2", "g3"],
        "game_date": pd.to_datetime(["2018-01-01", "2018-02-01", "2018-11-01"]),
        "home_team": "AAA", "away_team": "BBB",
        "player_id": ["p1", "p2", "p3"],
        "action_type": "Jump Shot", "basic_zone": "Mid-Range", "is_three": 0,
        "made": [1, 1, 1],
    })
    out = add_defense_history(df, expected_40pct())
    assert list(out["opponent"]) == ["BBB", "BBB", "BBB"]
    assert list(out["opponent_prior_shots"]) == [0, 1, 0]   # g2 sees g1; g3 is a new season, starts over
    assert np.allclose(out["opponent_prior_resid"], [0, 0.6, 0])


def test_opponent_defense_shrinks_toward_zero():
    df = pd.DataFrame({"opponent_prior_resid": [10.0, 10.0], "opponent_prior_shots": [100, 2000]})
    out = add_opponent_defense(df, k=900)
    assert np.allclose(out["opponent_defense"], [10 / 1000, 10 / 2900])
