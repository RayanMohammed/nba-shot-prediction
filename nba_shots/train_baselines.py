# Fit the simple model ladder (M0-M4) on the training seasons and score it on validation.
# Only the train and validation seasons are read; the covid and test seasons are never loaded here.
# Run from the project folder:   python -m nba_shots.train_baselines

import os

import pandas as pd

from nba_shots import config
from nba_shots.evaluate import calibration_table, fit_and_score, plot_calibration
from nba_shots.models import (predict_constant, predict_court_squares, predict_distance_only,
                              predict_distance_shape_zone, predict_logistic_all,
                              predict_logistic_detailed, predict_logistic_families_only,
                              predict_logistic_no_shot_type)
from nba_shots.splits import load_train_val

COLUMNS = ["season", "game_key", "game_date", "made", "shot_distance", "is_three", "basic_zone",
           "zone_name", "loc_x", "loc_y", "action_type", "quarter", "mins_left", "secs_left",
           "team_name", "home_team", "away_team"]

MODELS = {
    "M0 constant": predict_constant,
    "M1 distance only": predict_distance_only,
    "M2 distance shape + zone": predict_distance_shape_zone,
    "M3 court squares": predict_court_squares,
    "M4 logistic, families + flags": predict_logistic_all,
    "M4 families only": predict_logistic_families_only,
    "M4 detailed shot descriptions": predict_logistic_detailed,
    "M4 no shot type": predict_logistic_no_shot_type,
}

# models to also run on ALL training seasons (2003-04 on), as the comparison to the main setup
ALL_SEASON_COMPARISONS = ["M2 distance shape + zone", "M4 logistic, families + flags"]

# models drawn on the calibration plot (a constant has nothing to calibrate)
PLOT = ["M1 distance only", "M2 distance shape + zone", "M4 logistic, families + flags"]


def main():
    train, all_train, val = load_train_val(COLUMNS)

    rows = []
    calibration = {}
    for name, predict in MODELS.items():
        p = fit_and_score(name, predict, train, val, "main", rows)
        if name in PLOT:
            calibration[name] = calibration_table(val["made"], p)
    for name in ALL_SEASON_COMPARISONS:
        fit_and_score(name, MODELS[name], all_train, val, "all", rows)

    pd.DataFrame(rows).to_csv(os.path.join(config.RESULTS_DIR, "baselines_val.csv"), index=False)
    plot_calibration(calibration, os.path.join(config.RESULTS_DIR, "calibration_baselines_val.png"),
                     "Model ladder on validation (trained on 2012-13 to 2018-19)")


if __name__ == "__main__":
    main()
