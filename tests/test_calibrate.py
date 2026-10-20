import numpy as np
import pandas as pd

from nba_shots import config
from nba_shots.calibrate import (apply_platt, apply_smooth, fit_isotonic, fit_platt, fit_smooth,
                                 overfitting_checks)
from nba_shots.evaluate import score


def test_platt_and_isotonic_fix_predictions_that_are_too_low():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.2, 0.8, 40_000)
    y = (rng.uniform(0, 1, 40_000) < true_p).astype(int)
    too_low = true_p * 0.8  # a model that predicts 20% too low everywhere

    fit, check = slice(0, 20_000), slice(20_000, 40_000)  # learn on one half, check on the other
    platt = fit_platt(too_low[fit], y[fit])
    isotonic = fit_isotonic(too_low[fit], y[fit])

    before = score(y[check], too_low[check])["log_loss"]
    assert score(y[check], apply_platt(platt, too_low[check]))["log_loss"] < before
    assert score(y[check], isotonic.predict(too_low[check]))["log_loss"] < before


def test_overfitting_checks_run_and_report_both_directions():
    rng = np.random.default_rng(1)
    n = 4000
    p = rng.uniform(0.2, 0.8, n)
    preds = pd.DataFrame({
        "season": [config.CALIBRATION_FIT_SEASON] * (n // 2) + [config.CALIBRATION_CHECK_SEASON] * (n // 2),
        "game_key": ["g" + str(i // 50) for i in range(n)],
        "made": (rng.uniform(0, 1, n) < p).astype(int),
        config.MAIN_MODEL: p * 0.8,  # predictions 20% too low
    })
    checks = overfitting_checks(preds)
    assert len(checks) == 6                         # 3 methods x 2 directions
    assert (checks["gain_out_of_sample"] > 0).all() # fixing a real, consistent error helps on new shots


def test_smooth_fixes_a_bend_that_platt_cannot():
    rng = np.random.default_rng(2)
    n = 60_000
    true_p = rng.uniform(0.25, 0.75, n)
    y = (rng.uniform(0, 1, n) < true_p).astype(int)
    # a model that's right at the edges but too low in the middle (a bend, not a shift or stretch)
    bent = true_p - 0.08 * np.exp(-((true_p - 0.5) / 0.08) ** 2)

    fit, check = slice(0, n // 2), slice(n // 2, n)
    platt_loss = score(y[check], apply_platt(fit_platt(bent[fit], y[fit]), bent[check]))["log_loss"]
    smooth_loss = score(y[check], apply_smooth(fit_smooth(bent[fit], y[fit]), bent[check]))["log_loss"]
    assert smooth_loss < platt_loss
