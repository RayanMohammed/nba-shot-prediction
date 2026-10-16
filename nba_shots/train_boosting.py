# Fit gradient boosting (M5) on the training seasons and score it on validation, next to the best
# logistic regression (M4) for reference. Takes several minutes.
# Run from the project folder:   python -m nba_shots.train_boosting

import os

import pandas as pd

from nba_shots import config
from nba_shots.evaluate import calibration_table, fit_and_score, plot_calibration
from nba_shots.models import (predict_boosting, predict_boosting_detailed,
                              predict_boosting_families_only, predict_boosting_no_shot_type,
                              predict_logistic_all)
from nba_shots.splits import load_train_val
from nba_shots.train_baselines import COLUMNS

MODELS = {
    "M4 logistic, families + flags": predict_logistic_all,
    "M5 boosting, families + flags": predict_boosting,
    "M5 boosting, families only": predict_boosting_families_only,
    "M5 boosting, detailed shot descriptions": predict_boosting_detailed,
    "M5 boosting, no shot type": predict_boosting_no_shot_type,
}

PLOT = ["M4 logistic, families + flags", "M5 boosting, families + flags"]


def main():
    train, _, val = load_train_val(COLUMNS, include_old_seasons=False)

    rows = []
    calibration = {}
    for name, predict in MODELS.items():
        p = fit_and_score(name, predict, train, val, "main", rows)
        if name in PLOT:
            calibration[name] = calibration_table(val["made"], p)

    pd.DataFrame(rows).to_csv(os.path.join(config.RESULTS_DIR, "boosting_val.csv"), index=False)
    plot_calibration(calibration, os.path.join(config.RESULTS_DIR, "calibration_boosting_val.png"),
                     "Logistic vs boosting on validation")


if __name__ == "__main__":
    main()
