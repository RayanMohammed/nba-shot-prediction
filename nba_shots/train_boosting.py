# Fit gradient boosting (M5) on the training seasons and score it on validation. Takes several minutes.
# Run train_baselines first: the calibration plot here compares against M4's saved predictions.
# Run from the project folder:   python -m nba_shots.train_boosting

import os

import pandas as pd

from nba_shots import config
from nba_shots.evaluate import calibration_table, fit_and_score, plot_calibration, save_predictions
from nba_shots.models import (predict_boosting, predict_boosting_detailed,
                              predict_boosting_no_shot_type, predict_boosting_with_flags)
from nba_shots.splits import load_train_val
from nba_shots.train_baselines import COLUMNS

MODELS = {
    "M5 boosting, families": predict_boosting,
    "M5 boosting, families + flags": predict_boosting_with_flags,
    "M5 boosting, detailed shot descriptions": predict_boosting_detailed,
    "M5 boosting, no shot type": predict_boosting_no_shot_type,
}


def main():
    train, _, val = load_train_val(COLUMNS, include_old_seasons=False)

    rows = []
    predictions = {}
    for name, predict in MODELS.items():
        predictions[name] = fit_and_score(name, predict, train, val, "main", rows)

    pd.DataFrame(rows).to_csv(os.path.join(config.RESULTS_DIR, "boosting_val.csv"), index=False)
    os.makedirs(config.PREDICTIONS_DIR, exist_ok=True)
    save_predictions(val, predictions, os.path.join(config.PREDICTIONS_DIR, "boosting_val.parquet"))

    # calibration plot: best logistic (saved by train_baselines) vs the main boosting model
    calibration = {config.MAIN_MODEL: calibration_table(val["made"], predictions[config.MAIN_MODEL])}
    baseline_file = os.path.join(config.PREDICTIONS_DIR, "baselines_val.parquet")
    if os.path.exists(baseline_file):
        baselines = pd.read_parquet(baseline_file)
        calibration["M4 logistic, families"] = calibration_table(baselines["made"], baselines["M4 logistic, families"])
    plot_calibration(calibration, os.path.join(config.RESULTS_DIR, "calibration_boosting_val.png"),
                     "Logistic vs boosting on validation")


if __name__ == "__main__":
    main()
