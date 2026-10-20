import numpy as np
import pandas as pd

from nba_shots.models import (pick_heldout_games, predict_boosting, predict_constant, predict_court_squares,
                              stopping_set)


def test_constant_predicts_training_rate():
    train = pd.DataFrame({"made": [1, 0, 0, 0]})
    val = pd.DataFrame({"made": [1, 1, 0]})
    assert list(predict_constant(train, val)) == [0.25, 0.25, 0.25]


def test_court_squares_smoothing_and_unseen_squares():
    # training: 4 shots in one square (3 makes), 4 shots in another square (1 make) -> overall 0.5
    train = pd.DataFrame({"loc_x": [0.5] * 4 + [10.5] * 4, "loc_y": [6.0] * 8,
                          "made": [1, 1, 1, 0, 1, 0, 0, 0]})
    # val: first shot in the 3-of-4 square, second in a square with no training shots
    val = pd.DataFrame({"loc_x": [0.5, -20.0], "loc_y": [6.0, 30.0], "made": [1, 0]})
    p = predict_court_squares(train, val, size=2, k=4)
    assert np.isclose(p[0], (3 + 4 * 0.5) / (4 + 4))  # pulled from 0.75 toward 0.5
    assert np.isclose(p[1], 0.5)                       # no training shots there: overall rate


def fake_shots(n_games=20, shots_per_game=30, seed=0):
    """Made-up shots in the clean-data format, enough for the full feature + boosting pipeline."""
    rng = np.random.default_rng(seed)
    n = n_games * shots_per_game
    distance = rng.integers(0, 28, n)
    return pd.DataFrame({
        "season": "2018-19", "team_name": np.where(rng.uniform(0, 1, n) < 0.5, "Team A", "Team B"),
        "game_key": np.repeat(["g" + str(i) for i in range(n_games)], shots_per_game),
        "home_team": "AAA", "away_team": "BBB", "shot_distance": distance,
        "is_three": (distance >= 23).astype(int), "loc_x": rng.uniform(-20, 20, n),
        "loc_y": 5.25 + distance.astype(float), "mins_left": rng.integers(0, 12, n),
        "secs_left": rng.integers(0, 60, n), "quarter": rng.integers(1, 5, n),
        "basic_zone": np.where(distance < 4, "Restricted Area", "Mid-Range"), "zone_name": "Center",
        "action_type": np.where(distance < 4, "Layup Shot", "Jump Shot"),
        "made": (rng.uniform(0, 1, n) < np.where(distance < 4, 0.6, 0.35)).astype(int),
    })


def test_boosting_runs_end_to_end_on_tiny_data():
    train = fake_shots(seed=0)
    val = fake_shots(n_games=5, seed=1)
    p = predict_boosting(train, val)
    assert len(p) == len(val)
    assert ((p > 0) & (p < 1)).all()


def test_heldout_games_are_whole_games():
    games = pd.Series(np.repeat(["g" + str(i) for i in range(20)], 5))  # 20 games, 5 shots each
    held = pick_heldout_games(games, share=0.1)
    assert held.sum() == 10                            # 2 games x 5 shots
    assert games[held].nunique() == 2
    assert not set(games[held]) & set(games[~held])    # no game on both sides


def test_stopping_set_last_season_and_random_games():
    train = pd.DataFrame({"season": ["2016-17"] * 4 + ["2018-19"] * 3 + ["2017-18"] * 3,
                          "game_key": ["a", "a", "b", "b", "c", "c", "d", "e", "e", "f"]})
    assert list(stopping_set(train, "last_season")) == [False] * 4 + [True] * 3 + [False] * 3
    held = stopping_set(train, "random_games")
    assert not set(train["game_key"][held]) & set(train["game_key"][~held])  # whole games only


def test_boosting_runs_with_last_season_stop_and_no_xy():
    train = pd.concat([fake_shots(seed=0).assign(season="2017-18"),
                       fake_shots(seed=2).assign(season="2018-19")], ignore_index=True)
    train["game_key"] = train["season"] + "_" + train["game_key"]
    val = fake_shots(n_games=5, seed=1)
    p = predict_boosting(train, val, stopping="last_season", min_leaf=50, l2=1.0, with_xy=False)
    assert len(p) == len(val) and ((p > 0) & (p < 1)).all()


def test_last_season_stop_falls_back_when_only_one_season():
    train = pd.DataFrame({"season": "2018-19", "game_key": ["g" + str(i) for i in range(20)]})
    held = stopping_set(train, "last_season")
    assert 0 < held.sum() < len(train)  # some games held out, not all of them
