# Try a few boosting setups aimed at the era-overfitting we found, and score each on validation,
# season by season. Trains models, so run it on your own computer. Takes about 10 minutes.
# Run from the project folder:   python -m nba_shots.tune_boosting
#
# The winner gets written into config.py by hand (a visible, deliberate decision), then
# train_boosting / calibrate / compare_models / check_overfitting are re-run with it.

import os

import pandas as pd
from sklearn.metrics import log_loss

from nba_shots import config
from nba_shots.features import all_features, common_action_types
from nba_shots.models import fit_boosting, stopping_set
from nba_shots.splits import load_train_val
from nba_shots.train_baselines import COLUMNS

VARIANTS = {
    "A current (random-game stop)": {"stopping": "random_games", "min_leaf": 100, "l2": 0.0, "with_xy": True},
    "B last-season stop": {"stopping": "last_season", "min_leaf": 100, "l2": 0.0, "with_xy": True},
    "C last-season stop + cautious trees": {"stopping": "last_season", "min_leaf": 1000, "l2": 1.0, "with_xy": True},
    "D = C without raw x/y": {"stopping": "last_season", "min_leaf": 1000, "l2": 1.0, "with_xy": False},
}


def score_by_season(val, p, label, trees):
    """One row: log loss on each validation season and on both together."""
    row = {"variant": label, "trees": trees}
    for season in sorted(val["season"].unique()):
        in_season = (val["season"] == season).to_numpy()
        row["val " + season] = log_loss(val["made"].to_numpy()[in_season], p[in_season])
    row["val both"] = log_loss(val["made"], p)
    return row


def main():
    train, _, val = load_train_val(COLUMNS, include_old_seasons=False)
    common = common_action_types(train)

    rows = []
    for name, settings in VARIANTS.items():
        print(f"\n{name}: {settings}", flush=True)
        X_train = all_features(train, common, "family", with_xy=settings["with_xy"])
        X_val = all_features(val, common, "family", with_xy=settings["with_xy"]).reindex(columns=X_train.columns, fill_value=0)
        held = stopping_set(train, settings["stopping"])
        model, trees = fit_boosting(X_train, train["made"], held, settings["min_leaf"], settings["l2"])
        p = model.predict_proba(X_val.to_numpy())[:, 1]
        row = score_by_season(val, p, name, trees)
        rows.append(row)
        print("  " + " | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)

    # logistic regression reference, from the predictions train_baselines saved
    baseline_file = os.path.join(config.PREDICTIONS_DIR, "baselines_val.parquet")
    if os.path.exists(baseline_file):
        base = pd.read_parquet(baseline_file)
        rows.append(score_by_season(base, base["M4 logistic, families"].to_numpy(), "M4 logistic (reference)", None))

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(config.RESULTS_DIR, "boosting_tuning_val.csv"), index=False)
    print("\n" + results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
