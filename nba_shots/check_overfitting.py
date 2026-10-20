# Is the boosting model overfitting (memorizing its training shots)?
# Trains models, so run it on your own computer. Takes several minutes.
# Run from the project folder:   python -m nba_shots.check_overfitting
#
# We hold out 10% of the TRAINING games, grow the model tree by tree, and track log loss on:
#   1. the shots it trains on
#   2. the held-out training games (same seasons, never seen)
#   3. each validation season (later seasons, never seen)
# Gap between 1 and 2 = overfitting (memorizing). Gap between 2 and 3 = drift (the game changed).
# Logistic regression (M4) is measured the same way as a reference: a straight-line model can barely overfit.

import os

import matplotlib
matplotlib.use("Agg")  # save plots to files, no popup window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import log_loss

from nba_shots import config
from nba_shots.features import all_features, common_action_types
from nba_shots.models import fit_logistic, new_boosting_model, pick_heldout_games
from nba_shots.splits import load_train_val
from nba_shots.train_baselines import COLUMNS

SCORE_SAMPLE = 200_000  # score training shots on a random sample (scoring all 1.3M every step is slow)


def main():
    train, _, val = load_train_val(COLUMNS, include_old_seasons=False)
    held = pick_heldout_games(train["game_key"], config.BOOSTING_HELDOUT_GAMES)
    fit_part = train[~held]
    heldout_part = train[held]
    print("fit shots:", len(fit_part), "| held-out game shots:", len(heldout_part))

    # the same features as the main model, built for every set of shots
    common = common_action_types(fit_part)
    X_fit = all_features(fit_part, common, "family", with_xy=config.BOOSTING_WITH_XY)
    sets = {
        "training shots": fit_part.sample(n=SCORE_SAMPLE, random_state=0),
        "held-out training games": heldout_part,
        "val 2021-22": val[val["season"] == config.CALIBRATION_FIT_SEASON],
        "val 2022-23": val[val["season"] == config.CALIBRATION_CHECK_SEASON],
    }
    X_sets = {name: all_features(df, common, "family", with_xy=config.BOOSTING_WITH_XY).reindex(columns=X_fit.columns, fill_value=0).to_numpy()
              for name, df in sets.items()}
    y_sets = {name: df["made"].to_numpy() for name, df in sets.items()}

    # --- boosting: grow 50 trees at a time and score every set ---
    model = new_boosting_model(warm_start=True)
    X_fit_array = X_fit.to_numpy()
    y_fit = fit_part["made"].to_numpy()
    rows = []
    for n_trees in range(config.BOOSTING_TREES_STEP, config.OVERFIT_CHECK_MAX_TREES + 1, config.BOOSTING_TREES_STEP):
        model.set_params(max_iter=n_trees)
        model.fit(X_fit_array, y_fit)
        row = {"model": "M5 boosting", "trees": n_trees}
        for name in sets:
            row[name] = log_loss(y_sets[name], model.predict_proba(X_sets[name])[:, 1])
        rows.append(row)
        print(f"  {n_trees:4d} trees | " + " | ".join(f"{name} {row[name]:.4f}" for name in sets), flush=True)

    # --- logistic regression reference (no xy, same as M4) ---
    logistic_fit = fit_part.sample(n=min(len(fit_part), config.LOGISTIC_MAX_ROWS), random_state=0)
    X_log = all_features(logistic_fit, common, "family")
    logistic = fit_logistic(X_log, logistic_fit["made"])
    row = {"model": "M4 logistic", "trees": None}
    for name, df in sets.items():
        X = all_features(df, common, "family").reindex(columns=X_log.columns, fill_value=0)
        row[name] = log_loss(df["made"], logistic.predict_proba(X)[:, 1])
    rows.append(row)
    print("  M4 logistic | " + " | ".join(f"{name} {row[name]:.4f}" for name in sets))

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(config.RESULTS_DIR, "overfitting_check.csv"), index=False)

    # gaps at the tree count the main model actually uses (where held-out loss is lowest)
    boost = results[results["model"] == "M5 boosting"]
    best = boost.loc[boost["held-out training games"].idxmin()]
    print(f"\nAt {int(best['trees'])} trees (lowest held-out loss):")
    print(f"  overfitting gap (held-out games minus training shots): {best['held-out training games'] - best['training shots']:+.4f}")
    print(f"  drift gap (2022-23 minus held-out games):              {best['val 2022-23'] - best['held-out training games']:+.4f}")
    print(f"  M4 logistic overfitting gap, for reference:            {row['held-out training games'] - row['training shots']:+.4f}")

    # learning curve plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in sets:
        ax.plot(boost["trees"], boost[name], marker="o", label=name)
    ax.axvline(best["trees"], linestyle="--", color="gray", label="lowest held-out loss")
    ax.set_xlabel("number of trees")
    ax.set_ylabel("log loss (lower is better)")
    ax.set_title("M5 boosting: training vs unseen shots as trees are added")
    ax.legend()
    fig.savefig(os.path.join(config.RESULTS_DIR, "overfitting_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
