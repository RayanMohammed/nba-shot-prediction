import pytest

from nba_shots.leakage import check_features


def test_label_is_rejected():
    with pytest.raises(ValueError):
        check_features(["loc_x", "made"])


def test_post_game_columns_rejected_any_capitalization():
    with pytest.raises(ValueError):
        check_features(["FINAL_MARGIN"])


def test_normal_features_pass():
    check_features(["loc_x", "loc_y", "shot_distance", "action_type"])
