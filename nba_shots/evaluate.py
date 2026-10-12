# Scoring and calibration for any model's predictions.

import matplotlib
matplotlib.use("Agg")  # save plots to files, no popup window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def score(y, p):
    """The three numbers every model gets.

    log_loss: main metric. Punishes confident wrong answers hard (lower is better).
    brier: average squared gap between prediction and outcome (lower is better).
    auc: how well the model ranks makes above misses, ignoring calibration (higher is better).
    """
    return {"log_loss": log_loss(y, p), "brier": brier_score_loss(y, p), "auc": roc_auc_score(y, p)}


def calibration_table(y, p, n_bins=10):
    """Sort shots into n_bins equal-size groups by predicted probability, then compare
    the average prediction in each group with how often those shots actually went in."""
    df = pd.DataFrame({"pred": p, "made": y})
    df["group"] = pd.qcut(df["pred"], q=n_bins, duplicates="drop")
    table = df.groupby("group", observed=True).agg(avg_pred=("pred", "mean"),
                                                   actual_rate=("made", "mean"),
                                                   shots=("made", "size"))
    return table.reset_index(drop=True)


def plot_calibration(tables, path, title):
    """One line per model. A perfectly calibrated model sits on the dashed diagonal."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    for name, table in tables.items():
        ax.plot(table["avg_pred"], table["actual_rate"], marker="o", label=name)
    ax.set_xlabel("predicted make probability")
    ax.set_ylabel("actual make rate")
    ax.set_title(title)
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
