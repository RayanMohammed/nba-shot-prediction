# Does knowing WHO is shooting, or WHO is defending, help? Four models on validation:
#   main M5 | + shooter skill | + opponent defense | + both
# Each "skill" is tried with a few shrinkage values (k); the best on validation is used for "both".
# Trains 8 models, so run it on your own computer. Takes about 20 minutes.
# Run from the project folder:   python -m nba_shots.train_skill

import os

import pandas as pd

from nba_shots import config
from nba_shots.check_data import load_clean_data
from nba_shots.evaluate import bootstrap_log_loss_diff, fit_and_score, save_predictions, shot_log_loss
from nba_shots.features import (add_defense_history, add_opponent_defense, add_shooter_history,
                                add_shooter_skill, expected_make_rates)
from nba_shots.models import (predict_boosting, predict_boosting_with_both, predict_boosting_with_defense,
                              predict_boosting_with_skill)
from nba_shots.splits import assign_split, check_splits, seasons_in
from nba_shots.train_baselines import COLUMNS

SKILL_DATA_COLUMNS = COLUMNS + ["player_id"]
MAIN = "M5 boosting, families"
FEW_SHOOTER_SHOTS = 100   # "new" shooter: skill based on fewer earlier shots of this type
FEW_DEFENSE_SHOTS = 300   # "early season" for a defense: fewer earlier shots of this type faced (~10 games)


def load_with_history(include_test=False):
    """Every season from config.MAIN_TRAIN_START on (covid seasons too: they're real games that count as
    a player's history), with each shot's as-of shooter AND defense history added. The covid seasons are
    then dropped, since they're never trained or scored on. include_test=True is only for final_test.py."""
    all_seasons = [f.replace(".parquet", "") for f in sorted(os.listdir(config.CLEAN_DIR))]
    split_names = ["train", "covid", "val"] + (["test"] if include_test else [])
    seasons = [s for s in seasons_in(split_names, all_seasons) if s >= config.MAIN_TRAIN_START]
    df = assign_split(load_clean_data(config.CLEAN_DIR, columns=SKILL_DATA_COLUMNS, seasons=seasons))
    check_splits(df)
    expected = expected_make_rates(df[df["split"] == "train"])  # league averages from TRAINING seasons only
    df = add_shooter_history(df, expected)
    df = add_defense_history(df, expected)
    return df[df["split"] != "covid"]


def compare(val, predictions, a, b):
    """Game-level 95% interval for A's log loss minus B's (negative = A better)."""
    diff, low, high = bootstrap_log_loss_diff(val["made"], predictions[a], predictions[b], val["game_key"],
                                              n_boot=config.N_BOOTSTRAP)
    verdict = "A better" if high < 0 else ("B better" if low > 0 else "no clear difference")
    print(f"  {a:40s} vs {b:36s} {diff:+.4f} [{low:+.4f}, {high:+.4f}] {verdict}")
    return {"model_a": a, "model_b": b, "log_loss_diff": diff, "ci_low": low, "ci_high": high, "verdict": verdict}


def main():
    df = load_with_history()
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    del df
    print("train shots:", len(train), "| val shots:", len(val))

    rows = []
    predictions = {MAIN: fit_and_score(MAIN, predict_boosting, train, val, "main", rows)}

    # shooter skill: try each k, keep the best on validation
    shooter_loss = {}
    for k in config.SKILL_K_OPTIONS:
        name = f"M5 + shooter skill (k={k})"
        predictions[name] = fit_and_score(name, predict_boosting_with_skill, add_shooter_skill(train, k),
                                          add_shooter_skill(val, k), "main", rows)
        shooter_loss[k] = rows[-1]["log_loss"]
    best_shooter_k = min(shooter_loss, key=shooter_loss.get)  # the k with the lowest log loss

    # opponent defense: same
    defense_loss = {}
    for k in config.DEFENSE_K_OPTIONS:
        name = f"M5 + opponent defense (k={k})"
        predictions[name] = fit_and_score(name, predict_boosting_with_defense, add_opponent_defense(train, k),
                                          add_opponent_defense(val, k), "main", rows)
        defense_loss[k] = rows[-1]["log_loss"]
    best_defense_k = min(defense_loss, key=defense_loss.get)

    # both, each with its best k
    both = f"M5 + both (k={best_shooter_k}, {best_defense_k})"
    train_both = add_opponent_defense(add_shooter_skill(train, best_shooter_k), best_defense_k)
    val_both = add_opponent_defense(add_shooter_skill(val, best_shooter_k), best_defense_k)
    predictions[both] = fit_and_score(both, predict_boosting_with_both, train_both, val_both, "main", rows)

    pd.DataFrame(rows).to_csv(os.path.join(config.RESULTS_DIR, "skill_val.csv"), index=False)
    os.makedirs(config.PREDICTIONS_DIR, exist_ok=True)
    save_predictions(val, predictions, os.path.join(config.PREDICTIONS_DIR, "skill_val.parquet"))

    shooter = f"M5 + shooter skill (k={best_shooter_k})"
    defense = f"M5 + opponent defense (k={best_defense_k})"
    print("\nGame-level 95% intervals (A minus B, negative = A better):")
    comparisons = [compare(val, predictions, shooter, MAIN), compare(val, predictions, defense, MAIN),
                   compare(val, predictions, both, MAIN), compare(val, predictions, both, shooter),
                   compare(val, predictions, both, defense)]
    pd.DataFrame(comparisons).to_csv(os.path.join(config.RESULTS_DIR, "skill_comparisons_val.csv"), index=False)

    # who does each one help? new vs established shooters, early vs later-season defenses
    new_shooter = (val["shooter_prior_shots"] < FEW_SHOOTER_SHOTS).to_numpy()
    early_defense = (val["opponent_prior_shots"] < FEW_DEFENSE_SHOTS).to_numpy()
    group_rows = []
    for name in [MAIN, shooter, defense, both]:
        loss = shot_log_loss(val["made"], predictions[name])
        group_rows.append({"model": name,
                           "new shooters": loss[new_shooter].mean(), "established shooters": loss[~new_shooter].mean(),
                           "early-season defense": loss[early_defense].mean(), "later-season defense": loss[~early_defense].mean()})
    groups = pd.DataFrame(group_rows)
    groups.to_csv(os.path.join(config.RESULTS_DIR, "skill_by_group_val.csv"), index=False)
    print(f"\nLog loss by group ({new_shooter.mean():.1%} of val shots are by 'new' shooters, "
          f"{early_defense.mean():.1%} are against an 'early-season' defense):")
    print(groups.round(4).to_string(index=False))

    print(f"\nBest on validation: SKILL_K = {best_shooter_k}, DEFENSE_K = {best_defense_k} (write these into config.py)")


if __name__ == "__main__":
    main()
