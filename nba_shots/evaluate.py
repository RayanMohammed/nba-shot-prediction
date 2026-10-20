# Scoring and calibration for any model's predictions.

import matplotlib
matplotlib.use("Agg")  # save plots to files, no popup window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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


def fit_and_score(name, predict, train, val, train_seasons, rows):
    """Fit one model, print its validation scores, add them to rows, and return its predictions."""
    p = predict(train, val)
    s = score(val["made"], p)
    label = name if train_seasons == "main" else name + " (all seasons)"
    print(f"{label:44s} log loss {s['log_loss']:.4f} | brier {s['brier']:.4f} | auc {s['auc']:.4f}", flush=True)
    rows.append({"model": name, "train_seasons": train_seasons, "log_loss": s["log_loss"],
                 "brier": s["brier"], "auc": s["auc"]})
    return p


def save_predictions(val, predictions, path):
    """Save each model's validation predictions next to the shot's season, game and outcome.

    predictions: dict of model name -> array of P(make), one value per row of val.
    """
    out = val[["season", "game_key", "made"]].reset_index(drop=True)
    for name, p in predictions.items():
        out[name] = p
    out.to_parquet(path, index=False)


def logit(p):
    """Probability -> log-odds. Clipped so 0 and 1 don't turn into minus/plus infinity."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_slope_intercept(y, p):
    """Fit outcome ~ logit(prediction). A perfectly calibrated model gets slope 1, intercept 0.

    slope < 1: predictions too extreme (overconfident); slope > 1: too timid.
    intercept > 0: predictions too low overall; intercept < 0: too high overall.
    """
    model = LogisticRegression(C=1e6)  # huge C = essentially no regularization
    model.fit(logit(np.asarray(p)).reshape(-1, 1), y)
    return model.coef_[0][0], model.intercept_[0]


def expected_calibration_error(y, p, n_bins=10):
    """Average gap between predicted and actual make rate over n_bins equal-size groups of shots."""
    table = calibration_table(y, p, n_bins)
    gaps = (table["avg_pred"] - table["actual_rate"]).abs()
    return (gaps * table["shots"]).sum() / table["shots"].sum()


def full_report(y, p):
    """score() plus the calibration numbers."""
    report = score(y, p)
    slope, intercept = calibration_slope_intercept(y, p)
    report["calib_slope"] = slope
    report["calib_intercept"] = intercept
    report["ece"] = expected_calibration_error(y, p)
    report["avg_pred"] = float(np.mean(p))
    report["actual_rate"] = float(np.mean(y))
    return report


def shot_log_loss(y, p):
    """Log loss of each shot separately (log_loss() gives only the average)."""
    y = np.asarray(y)
    p = np.clip(np.asarray(p), 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def bootstrap_log_loss_diff(y, p_a, p_b, games, n_boot=1000, seed=0):
    """Is model A really better than model B, or just lucky on these games?

    Paired, game-level bootstrap: resample whole GAMES with replacement (shots in one game are
    related, so resampling single shots would make us overconfident), and recompute
    "A's log loss minus B's log loss" each time. Negative = A is better.
    Returns (difference on the real data, 2.5th percentile, 97.5th percentile).
    If the interval doesn't include 0, the difference is unlikely to be luck.
    """
    diff = shot_log_loss(y, p_a) - shot_log_loss(y, p_b)
    per_game = pd.DataFrame({"game": np.asarray(games), "diff": diff}).groupby("game")["diff"].agg(["sum", "count"])
    sums = per_game["sum"].to_numpy()
    counts = per_game["count"].to_numpy()

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(sums), size=len(sums))  # game numbers, picked with replacement
        boot.append(sums[pick].sum() / counts[pick].sum())
    low, high = np.percentile(boot, [2.5, 97.5])
    return sums.sum() / counts.sum(), low, high
