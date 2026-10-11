import pandas as pd
import pytest

from nba_shots.splits import assign_split, check_splits

SPLITS = {"train": ("2001-02", "2002-03"), "val": ("2003-04", "2003-04"), "test": ("2004-05", "2004-05")}
ORDER = ["train", "val", "test"]


def fake_games(rows):
    """rows: list of (season, game_key, game_date)."""
    df = pd.DataFrame(rows, columns=["season", "game_key", "game_date"])
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def test_seasons_go_to_the_right_split():
    df = fake_games([("2001-02", "g1", "2002-01-01"), ("2003-04", "g2", "2004-01-01"),
                     ("2004-05", "g3", "2005-01-01"), ("1999-00", "g0", "2000-01-01")])
    assert list(assign_split(df, SPLITS)["split"]) == ["train", "val", "test", "unused"]


def test_good_split_passes():
    df = fake_games([("2002-03", "g1", "2003-04-01"), ("2003-04", "g2", "2003-11-01"),
                     ("2004-05", "g3", "2004-11-01")])
    check_splits(assign_split(df, SPLITS), ORDER)


def test_game_in_two_splits_fails():
    df = fake_games([("2002-03", "g1", "2003-04-01"), ("2003-04", "g1", "2003-04-01")])
    with pytest.raises(ValueError):
        check_splits(assign_split(df, SPLITS), ORDER)


def test_val_game_before_train_ends_fails():
    df = fake_games([("2002-03", "g1", "2004-06-01"), ("2003-04", "g2", "2003-11-01")])
    with pytest.raises(ValueError):
        check_splits(assign_split(df, SPLITS), ORDER)


def test_season_in_two_splits_fails():
    bad = {"train": ("2001-02", "2003-04"), "val": ("2003-04", "2003-04")}
    with pytest.raises(ValueError):
        assign_split(fake_games([("2003-04", "g1", "2004-01-01")]), bad)
