import pandas as pd
import pytest

from nba_shots import compare_models, config


def test_prediction_files_must_line_up(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PREDICTIONS_DIR", str(tmp_path))  # use a temp folder for this test
    shots = {"season": ["2021-22"] * 2, "game_key": ["g1", "g2"], "made": [1, 0]}
    pd.DataFrame({**shots, "M0 constant": [0.5, 0.5]}).to_parquet(tmp_path / "baselines_val.parquet")
    pd.DataFrame({**shots, "M5 boosting, families": [0.7, 0.2]}).to_parquet(tmp_path / "boosting_val.parquet")
    both = compare_models.load_all_predictions()
    assert list(both.columns) == ["season", "game_key", "made", "M0 constant", "M5 boosting, families"]

    # same shots in a different order -> refuse
    flipped = {"season": ["2021-22"] * 2, "game_key": ["g2", "g1"], "made": [0, 1], "M5 boosting, families": [0.2, 0.7]}
    pd.DataFrame(flipped).to_parquet(tmp_path / "boosting_val.parquet")
    with pytest.raises(ValueError):
        compare_models.load_all_predictions()
