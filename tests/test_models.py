import numpy as np
import pandas as pd

from nba_shots.models import predict_constant, predict_court_squares


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
