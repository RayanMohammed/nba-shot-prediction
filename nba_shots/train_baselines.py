# Fit the model ladder on the training seasons and score it on validation.
# Only the train and validation seasons are read; the covid and test seasons are never loaded here.
# Run from the project folder:   python -m nba_shots.train_baselines

import os

import pandas as pd

from nba_shots import config
from nba_shots.check_data import load_clean_data
from nba_shots.evaluate import calibration_table, plot_calibration, score
from nba_shots.models import (predict_constant, predict_court_squares, predict_distance_only,
                              predict_distance_shape_zone, predict_logistic_all,
                              predict_logistic_no_action)
from nba_shots.splits import assign_split, check_splits, seasons_in

COLUMNS = ["season", "game_key", "game_date", "made", "shot_distance", "is_three", "basic_zone",
           "zone_name", "loc_x", "loc_y", "action_type", "quarter", "mins_left", "secs_left",
           "team_name", "home_team", "away_team"]

MODELS = {
    "M0 constant": predict_constant,
    "M1 distance only": predict_distance_only,
    "M2 distance shape + zone": predict_distance_shape_zone,
    "M3 court squares": predict_court_squares,
    "M4 logistic, all features": predict_logistic_all,
    "M4 without action type": predict_logistic_no_action,
}

# models to also re-run trained on recent seasons only
RECENT_EXPERIMENTS = ["M2 distance shape + zone", "M4 logistic, all features"]

# models drawn on the calibration plot (a constant has nothing to calibrate)
PLOT = ["M1 distance only", "M2 distance shape + zone", "M4 logistic, all features"]


def run(name, predict, train, val, train_seasons, rows, calibration):
    """Fit one model, print and store its validation scores."""
    p = predict(train, val)
    s = score(val["made"], p)
    label = name if train_seasons == "all" else name + " (recent only)"
    print(f"{label:42s} log loss {s['log_loss']:.4f} | brier {s['brier']:.4f} | auc {s['auc']:.4f}", flush=True)
    rows.append({"model": name, "train_seasons": train_seasons, "log_loss": s["log_loss"],
                 "brier": s["brier"], "auc": s["auc"]})
    if name in PLOT and train_seasons == "all":
        calibration[name] = calibration_table(val["made"], p)


def main():
    all_seasons = [f.replace(".parquet", "") for f in sorted(os.listdir(config.CLEAN_DIR))]
    seasons = seasons_in(["train", "val"], all_seasons)
    df = assign_split(load_clean_data(config.CLEAN_DIR, columns=COLUMNS, seasons=seasons))
    check_splits(df)
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    del df  # train and val are copies, so free the full table to save memory
    print("train shots:", len(train), "| val shots:", len(val))

    rows = []
    calibration = {}
    for name, predict in MODELS.items():
        run(name, predict, train, val, "all", rows, calibration)
    recent_train = train[train["season"] >= config.RECENT_TRAIN_START]
    print("recent train shots:", len(recent_train))
    for name in RECENT_EXPERIMENTS:
        run(name, MODELS[name], recent_train, val, config.RECENT_TRAIN_START + " on", rows, calibration)

    pd.DataFrame(rows).to_csv(os.path.join(config.RESULTS_DIR, "baselines_val.csv"), index=False)
    plot_calibration(calibration, os.path.join(config.RESULTS_DIR, "calibration_baselines_val.png"),
                     "Model ladder on validation (2021-22, 2022-23)")


if __name__ == "__main__":
    main()
