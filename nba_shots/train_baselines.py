# Fit the baseline ladder on the training seasons and score it on validation.
# The test season is never loaded here.
# Run from the project folder:   python -m nba_shots.train_baselines

import os

import pandas as pd

from nba_shots import config
from nba_shots.check_data import load_clean_data
from nba_shots.evaluate import calibration_table, plot_calibration, score
from nba_shots.models import (predict_constant, predict_court_squares, predict_distance_only,
                              predict_distance_shape_zone)
from nba_shots.splits import assign_split, check_splits

COLUMNS = ["season", "game_key", "game_date", "made", "shot_distance", "is_three",
           "basic_zone", "loc_x", "loc_y"]

MODELS = {
    "M0 constant": predict_constant,
    "M1 distance only": predict_distance_only,
    "M2 distance shape + zone": predict_distance_shape_zone,
    "M3 court squares": predict_court_squares,
}


def main():
    df = assign_split(load_clean_data(config.CLEAN_DIR, columns=COLUMNS))
    check_splits(df)
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    recent_train = train[train["season"] >= config.RECENT_TRAIN_START]
    print("train shots:", len(train), "| recent train shots:", len(recent_train), "| val shots:", len(val))

    rows = []
    calibration = {}
    for name, predict in MODELS.items():
        p = predict(train, val)
        s = score(val["made"], p)
        print(f"{name:32s} log loss {s['log_loss']:.4f} | brier {s['brier']:.4f} | auc {s['auc']:.4f}")
        rows.append({"model": name, "train_seasons": "all", "log_loss": s["log_loss"],
                     "brier": s["brier"], "auc": s["auc"]})
        if name != "M0 constant":  # a constant has nothing to calibrate
            calibration[name] = calibration_table(val["made"], p)

    # experiment: same model as M2, trained only on recent seasons
    p = predict_distance_shape_zone(recent_train, val)
    s = score(val["made"], p)
    print(f"{'M2, recent seasons only':32s} log loss {s['log_loss']:.4f} | brier {s['brier']:.4f} | auc {s['auc']:.4f}")
    rows.append({"model": "M2 distance shape + zone", "train_seasons": config.RECENT_TRAIN_START + " on",
                 "log_loss": s["log_loss"], "brier": s["brier"], "auc": s["auc"]})

    pd.DataFrame(rows).to_csv(os.path.join(config.RESULTS_DIR, "baselines_val.csv"), index=False)
    plot_calibration(calibration, os.path.join(config.RESULTS_DIR, "calibration_baselines_val.png"),
                     "Baselines on validation (2021-22, 2022-23)")


if __name__ == "__main__":
    main()
