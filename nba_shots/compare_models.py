# Are the gaps between models real? Game-level paired bootstrap on the saved validation predictions.
# No training here: run train_baselines and train_boosting first. Takes under a minute.
# Run from the project folder:   python -m nba_shots.compare_models

import os

import pandas as pd

from nba_shots import config
from nba_shots.evaluate import bootstrap_log_loss_diff

# (model A, model B): each row reports A's log loss minus B's. Negative = A is better.
PAIRS = [
    ("M1 distance only", "M0 constant"),
    ("M2 distance shape + zone", "M1 distance only"),
    ("M3 court squares", "M2 distance shape + zone"),
    ("M4 logistic, families", "M2 distance shape + zone"),
    ("M4 logistic, no shot type", "M2 distance shape + zone"),
    ("M5 boosting, families", "M4 logistic, families"),
    ("M5 boosting, no shot type", "M5 boosting, families"),
    ("M5 boosting, families + flags", "M5 boosting, families"),
    ("M5 boosting, detailed shot descriptions", "M5 boosting, families"),
    ("M4 logistic, families (all seasons)", "M4 logistic, families"),
    ("M2 distance shape + zone (all seasons)", "M2 distance shape + zone"),
]


def load_all_predictions():
    """Put the baseline and boosting prediction files side by side (same validation shots, same order)."""
    baselines = pd.read_parquet(os.path.join(config.PREDICTIONS_DIR, "baselines_val.parquet"))
    boosting = pd.read_parquet(os.path.join(config.PREDICTIONS_DIR, "boosting_val.parquet"))
    same_shots = (baselines["game_key"].equals(boosting["game_key"]) and baselines["made"].equals(boosting["made"]))
    if not same_shots:
        raise ValueError("The two prediction files don't have the same validation shots in the same order")
    model_cols = [c for c in boosting.columns if c not in ["season", "game_key", "made"]]
    return pd.concat([baselines, boosting[model_cols]], axis=1)


def main():
    preds = load_all_predictions()
    rows = []
    for a, b in PAIRS:
        diff, low, high = bootstrap_log_loss_diff(preds["made"], preds[a], preds[b], preds["game_key"],
                                                  n_boot=config.N_BOOTSTRAP)
        verdict = "A better" if high < 0 else ("B better" if low > 0 else "no clear difference")
        print(f"{a:42s} vs {b:28s} {diff:+.4f}  [{low:+.4f}, {high:+.4f}]  {verdict}", flush=True)
        rows.append({"model_a": a, "model_b": b, "log_loss_diff": diff, "ci_low": low, "ci_high": high,
                     "verdict": verdict})
    pd.DataFrame(rows).to_csv(os.path.join(config.RESULTS_DIR, "model_comparisons_val.csv"), index=False)


if __name__ == "__main__":
    main()
