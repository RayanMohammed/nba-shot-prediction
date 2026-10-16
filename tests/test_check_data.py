import pandas as pd

from nba_shots.check_data import load_clean_data, run_checks


def fake_clean(rows):
    """rows: list of (loc_y, game_date) pairs; everything else is filled with normal values."""
    return pd.DataFrame({
        "season": "2018-19", "game_key": "g1", "home_team": "AAA", "player_id": "1", "quarter": 1,
        "loc_x": 0.0, "loc_y": [r[0] for r in rows], "shot_distance": 12, "is_three": 0,
        "made": 1, "game_date": pd.to_datetime([r[1] for r in rows]),
    })


def test_clean_data_passes():
    _, problems = run_checks(fake_clean([(17.25, "2019-01-01"), (17.25, "2019-01-01")]))
    assert problems == []


def test_wrong_coordinate_scale_is_caught():
    # y = 6.45 means 1.2 ft from the hoop, but the distance column says 12 ft
    _, problems = run_checks(fake_clean([(6.45, "2019-01-01")]))
    assert any("coordinate" in p for p in problems)


def test_game_with_two_dates_is_caught():
    _, problems = run_checks(fake_clean([(17.25, "2019-01-01"), (17.25, "2019-01-02")]))
    assert any("game_key" in p for p in problems)



def test_load_only_requested_seasons_and_columns(tmp_path):
    for season in ["2001-02", "2002-03"]:
        pd.DataFrame({"season": [season], "made": [1], "extra": ["x"]}).to_parquet(tmp_path / (season + ".parquet"))
    df = load_clean_data(tmp_path, columns=["season", "made"], seasons=["2002-03"])
    assert list(df["season"]) == ["2002-03"]
    assert list(df.columns) == ["season", "made"]
