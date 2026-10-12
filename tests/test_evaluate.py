import numpy as np

from nba_shots.evaluate import score


def test_coin_flip_scores():
    # predicting 50% on every shot: log loss is ln(2), brier is 0.25, auc is 0.5
    s = score([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5])
    assert np.isclose(s["log_loss"], np.log(2))
    assert np.isclose(s["brier"], 0.25)
    assert np.isclose(s["auc"], 0.5)
