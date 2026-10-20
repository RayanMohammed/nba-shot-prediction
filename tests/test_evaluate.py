import numpy as np
import pandas as pd

from nba_shots.evaluate import (bootstrap_log_loss_diff, calibration_slope_intercept,
                                expected_calibration_error, save_predictions, score)


def fake_outcomes(n=50_000, seed=0):
    """Predictions spread from 20% to 80%, and outcomes that really happen at those rates."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.2, 0.8, n)
    y = (rng.uniform(0, 1, n) < p).astype(int)
    return p, y


def test_coin_flip_scores():
    # predicting 50% on every shot: log loss is ln(2), brier is 0.25, auc is 0.5
    s = score([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5])
    assert np.isclose(s["log_loss"], np.log(2))
    assert np.isclose(s["brier"], 0.25)
    assert np.isclose(s["auc"], 0.5)


def test_calibrated_predictions_get_slope_1_intercept_0():
    p, y = fake_outcomes()
    slope, intercept = calibration_slope_intercept(y, p)
    assert abs(slope - 1) < 0.1
    assert abs(intercept) < 0.05
    assert expected_calibration_error(y, p) < 0.02


def test_overconfident_predictions_get_slope_below_1():
    p, y = fake_outcomes()
    too_extreme = 1 / (1 + np.exp(-2 * np.log(p / (1 - p))))  # double every log-odds
    slope, _ = calibration_slope_intercept(y, too_extreme)
    assert slope < 0.7


def test_bootstrap_identical_models_and_clearly_better_model():
    p, y = fake_outcomes(n=20_000)
    games = np.arange(len(y)) // 100  # 200 fake games of 100 shots
    diff, low, high = bootstrap_log_loss_diff(y, p, p, games, n_boot=200)
    assert diff == 0 and low == 0 and high == 0
    diff, low, high = bootstrap_log_loss_diff(y, p, np.full(len(y), 0.5), games, n_boot=200)
    assert high < 0  # the true probabilities beat a coin flip, and the interval says so


def test_save_predictions(tmp_path):
    val = pd.DataFrame({"season": ["2021-22", "2021-22"], "game_key": ["g1", "g2"], "made": [1, 0],
                        "extra": [5, 6]}, index=[10, 20])
    save_predictions(val, {"model A": np.array([0.6, 0.3])}, tmp_path / "p.parquet")
    out = pd.read_parquet(tmp_path / "p.parquet")
    assert list(out.columns) == ["season", "game_key", "made", "model A"]
    assert list(out["model A"]) == [0.6, 0.3]


def test_ece_of_a_constant_prediction_is_its_overall_gap():
    assert np.isclose(expected_calibration_error([1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]), 0.25)
