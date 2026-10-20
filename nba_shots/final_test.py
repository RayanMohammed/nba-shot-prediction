# THE FINAL TEST: the only script that ever reads the test season (2023-24).
# Run it once, after every modeling decision is made. Whatever it says is the result: no tuning afterwards.
# Trains models, so run it on your own computer. Takes about 15 minutes.
# Run from the project folder:   python -m nba_shots.final_test
#
# For each of the four models (main, + shooter skill, + opponent defense, + both):
#   1. train on 2012-13..2018-19 + 2021-22 (most recent data; the stopping referee is 2021-22)
#   2. learn the calibrator (config.CALIBRATION_METHOD) on its 2022-23 predictions
#   3. score 2023-24 once
# Plus M0 / M2 / M4 references trained the same way, and the main + both models saved for the app.

import os
import pickle

import pandas as pd

from nba_shots import config
from nba_shots.calibrate import apply_calibrator, fit_calibrator
from nba_shots.evaluate import (bootstrap_log_loss_diff, calibration_table, full_report,
                                plot_calibration, save_predictions, shot_log_loss)
from nba_shots.features import add_opponent_defense, add_shooter_skill, all_features, common_action_types, shot_family
from nba_shots.models import (DEFENSE_COLUMNS, SKILL_COLUMNS, fit_boosting, predict_constant,
                              predict_distance_shape_zone, predict_logistic_all, stopping_set)
from nba_shots.train_skill import FEW_DEFENSE_SHOTS, FEW_SHOOTER_SHOTS, load_with_history

FOUR_MODELS = {  # name -> extra columns on top of the main model
    "main": None,
    "+ shooter skill": SKILL_COLUMNS,
    "+ opponent defense": DEFENSE_COLUMNS,
    "+ both": SKILL_COLUMNS + DEFENSE_COLUMNS,
}

# (model A, model B): A's log loss minus B's on 2023-24. Negative = A is better.
PAIRS = [
    ("main", "main, uncalibrated"),
    ("main", "M4 logistic, families"),
    ("M4 logistic, families", "M2 distance shape + zone"),
    ("+ shooter skill", "main"),
    ("+ opponent defense", "main"),
    ("+ both", "main"),
    ("+ both", "+ shooter skill"),
    ("+ both", "+ opponent defense"),
]


def load_final_data():
    """2012-13..2018-19 + 2021-22 for training, 2022-23 for calibration, 2023-24 for the test,
    with as-of shooter and defense skill already on every shot."""
    df = load_with_history(include_test=True)
    df = add_opponent_defense(add_shooter_skill(df, config.SKILL_K), config.DEFENSE_K)
    final_train = df[(df["split"] == "train") | (df["season"] == config.CALIBRATION_FIT_SEASON)]
    calib = df[df["season"] == config.CALIBRATION_CHECK_SEASON]
    test = df[df["split"] == "test"]
    print("final training shots:", len(final_train), "| calibration shots:", len(calib), "| TEST shots:", len(test))
    return final_train, calib, test


def fit_final_model(final_train, calib, test, extra_cols):
    """Train one boosting model, calibrate it on calib, and predict test.
    Returns (uncalibrated test predictions, calibrated test predictions, bundle for saving)."""
    common = common_action_types(final_train)
    X_train = all_features(final_train, common, "family", with_xy=config.BOOSTING_WITH_XY, extra_cols=extra_cols)

    def features_for(df):
        X = all_features(df, common, "family", with_xy=config.BOOSTING_WITH_XY, extra_cols=extra_cols)
        return X.reindex(columns=X_train.columns, fill_value=0).to_numpy()

    model, trees = fit_boosting(X_train, final_train["made"], stopping_set(final_train))
    calibrator = fit_calibrator(config.CALIBRATION_METHOD, model.predict_proba(features_for(calib))[:, 1], calib["made"])
    p_raw = model.predict_proba(features_for(test))[:, 1]
    p_cal = apply_calibrator(config.CALIBRATION_METHOD, calibrator, p_raw)
    bundle = {
        "model": model,
        "calibration_method": config.CALIBRATION_METHOD,
        "calibrator": calibrator,
        "common_actions": common,
        "feature_columns": list(X_train.columns),
        "extra_cols": extra_cols,
        "skill_k": config.SKILL_K,
        "defense_k": config.DEFENSE_K,
        "trees": trees,
        "trained_on_seasons": sorted(final_train["season"].unique()),
        "calibrated_on_season": config.CALIBRATION_CHECK_SEASON,
    }
    return p_raw, p_cal, bundle


def main():
    print("Opening the test season. Results from here on are final.\n")
    final_train, calib, test = load_final_data()

    # --- the four models (each calibrated), plus references ---
    predictions = {
        "M0 constant": predict_constant(final_train, test),
        "M2 distance shape + zone": predict_distance_shape_zone(final_train, test),
        "M4 logistic, families": predict_logistic_all(final_train, test),
    }
    bundles = {}
    for name, extra_cols in FOUR_MODELS.items():
        print(f"\ntraining: {name}")
        p_raw, p_cal, bundles[name] = fit_final_model(final_train, calib, test, extra_cols)
        predictions[name] = p_cal
        if name == "main":
            predictions["main, uncalibrated"] = p_raw  # to see what calibration did on the test

    # --- scores ---
    rows = []
    for name, p in predictions.items():
        r = full_report(test["made"], p)
        r["model"] = name
        rows.append(r)
    report = pd.DataFrame(rows)[["model", "log_loss", "brier", "auc", "calib_slope", "calib_intercept",
                                 "ece", "avg_pred", "actual_rate"]]
    report.to_csv(os.path.join(config.RESULTS_DIR, "final_test.csv"), index=False)
    print("\n" + report.round(4).to_string(index=False))

    print("\nGame-level 95% intervals (A minus B, negative = A better):")
    comparison_rows = []
    for a, b in PAIRS:
        diff, low, high = bootstrap_log_loss_diff(test["made"], predictions[a], predictions[b], test["game_key"],
                                                  n_boot=config.N_BOOTSTRAP)
        verdict = "A better" if high < 0 else ("B better" if low > 0 else "no clear difference")
        print(f"  {a:24s} vs {b:26s} {diff:+.4f} [{low:+.4f}, {high:+.4f}] {verdict}")
        comparison_rows.append({"model_a": a, "model_b": b, "log_loss_diff": diff, "ci_low": low,
                                "ci_high": high, "verdict": verdict})
    pd.DataFrame(comparison_rows).to_csv(os.path.join(config.RESULTS_DIR, "final_test_comparisons.csv"), index=False)

    # --- who does each model help? (same groups as on validation) ---
    new_shooter = (test["shooter_prior_shots"] < FEW_SHOOTER_SHOTS).to_numpy()
    early_defense = (test["opponent_prior_shots"] < FEW_DEFENSE_SHOTS).to_numpy()
    group_rows = []
    for name in FOUR_MODELS:
        loss = shot_log_loss(test["made"], predictions[name])
        group_rows.append({"model": name,
                           "new shooters": loss[new_shooter].mean(), "established shooters": loss[~new_shooter].mean(),
                           "early-season defense": loss[early_defense].mean(), "later-season defense": loss[~early_defense].mean()})
    groups = pd.DataFrame(group_rows)
    groups.to_csv(os.path.join(config.RESULTS_DIR, "final_test_by_group.csv"), index=False)
    print("\nLog loss by group:")
    print(groups.round(4).to_string(index=False))

    # --- where are the main and full models right or wrong? one row per shot family ---
    families = test["action_type"].map({a: shot_family(a) for a in test["action_type"].unique()}).to_numpy()
    family_tables = []
    for name in ["main", "+ both"]:
        t = pd.DataFrame({"family": families, "made": test["made"].to_numpy(), "pred": predictions[name],
                          "loss": shot_log_loss(test["made"], predictions[name])})
        t = t.groupby("family").agg(shots=("made", "size"), actual_rate=("made", "mean"),
                                    avg_pred=("pred", "mean"), log_loss=("loss", "mean"))
        t["gap"] = t["avg_pred"] - t["actual_rate"]
        t["model"] = name
        family_tables.append(t.reset_index())
    by_family = pd.concat(family_tables).sort_values(["model", "shots"], ascending=[False, False])
    by_family.to_csv(os.path.join(config.RESULTS_DIR, "final_test_by_family.csv"), index=False)
    print("\nBy shot family (gap = predicted minus actual):")
    print(by_family.round(4).to_string(index=False))

    tables = {name: calibration_table(test["made"], predictions[name])
              for name in ["M4 logistic, families", "main, uncalibrated", "main", "+ both"]}
    plot_calibration(tables, os.path.join(config.RESULTS_DIR, "calibration_final_test.png"),
                     "Final test: 2023-24 (never used before)")

    os.makedirs(config.PREDICTIONS_DIR, exist_ok=True)
    save_predictions(test, predictions, os.path.join(config.PREDICTIONS_DIR, "test.parquet"))

    # --- save the two models the app can use: shot only, and shot + who's shooting + who's defending ---
    with open(config.FINAL_MODEL_PATH, "wb") as f:  # "wb" = write bytes (pickle files aren't text)
        pickle.dump({"main": bundles["main"], "+ both": bundles["+ both"]}, f)
    print("\nsaved models to", config.FINAL_MODEL_PATH)


if __name__ == "__main__":
    main()
