# Did crowds matter? Train the main model on normal seasons (2012-13..2018-19, all with fans), then score
# the covid seasons and see where real shooting differs from what the model expects:
#   gap = actual make rate - predicted (positive = shooters did better than normal-era basketball predicts)
#   home edge beyond model = how much more home shooters beat the model than away shooters do
# If crowds help the home team, the home edge should shrink when there are no fans.
# Trains a model, so run it on your own computer. Takes a few minutes.
# Run from the project folder:   python -m nba_shots.covid_analysis

import os

import matplotlib
matplotlib.use("Agg")  # save plots to files, no popup window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nba_shots import config
from nba_shots.check_data import load_clean_data
from nba_shots.features import add_is_home
from nba_shots.models import predict_boosting
from nba_shots.splits import assign_split
from nba_shots.train_baselines import COLUMNS


def per_game_table(df, pred):
    """One row per game: made-minus-predicted totals and shot counts, for home and away shooters."""
    t = pd.DataFrame({"game": df["game_key"].to_numpy(), "home": df["is_home"].to_numpy(),
                      "resid": df["made"].to_numpy() - pred})
    t["resid_home"] = np.where(t["home"] == 1, t["resid"], 0.0)
    t["resid_away"] = np.where(t["home"] == 0, t["resid"], 0.0)
    t["n_home"] = (t["home"] == 1).astype(int)
    t["n_away"] = (t["home"] == 0).astype(int)
    return t.groupby("game")[["resid", "resid_home", "resid_away", "n_home", "n_away"]].sum()


def gap_and_home_edge(games):
    """From per-game totals: overall gap (actual - predicted) and home edge beyond the model."""
    n = games["n_home"].sum() + games["n_away"].sum()
    gap = games["resid"].sum() / n
    home_edge = games["resid_home"].sum() / games["n_home"].sum() - games["resid_away"].sum() / games["n_away"].sum()
    return gap, home_edge


def bootstrap_period(games, n_boot, seed=0):
    """95% intervals for the gap and the home edge, resampling whole games."""
    rng = np.random.default_rng(seed)
    gaps, edges = [], []
    for _ in range(n_boot):
        sample = games.iloc[rng.integers(0, len(games), size=len(games))]
        gap, edge = gap_and_home_edge(sample)
        gaps.append(gap)
        edges.append(edge)
    return np.percentile(gaps, [2.5, 97.5]), np.percentile(edges, [2.5, 97.5])


def main():
    seasons = [f.replace(".parquet", "") for f in sorted(os.listdir(config.CLEAN_DIR))]
    seasons = [s for s in seasons if config.MAIN_TRAIN_START <= s <= "2021-22"]  # never loads 2022-23 or the test season
    df = assign_split(load_clean_data(config.CLEAN_DIR, columns=COLUMNS, seasons=seasons))
    train = df[df["split"] == "train"]
    later = add_is_home(df[df["split"] != "train"])  # covid seasons + 2021-22
    del df
    print("training shots (normal seasons):", len(train), "| scored shots:", len(later))

    pred = predict_boosting(train, later)

    rows = []
    for label, start, end in config.COVID_PERIODS:
        in_period = ((later["game_date"] >= start) & (later["game_date"] <= end)).to_numpy()
        part = later[in_period]
        games = per_game_table(part, pred[in_period])
        gap, edge = gap_and_home_edge(games)
        (gap_low, gap_high), (edge_low, edge_high) = bootstrap_period(games, config.N_BOOTSTRAP)
        rows.append({"period": label, "games": len(games), "shots": len(part),
                     "actual_rate": part["made"].mean(), "predicted_rate": pred[in_period].mean(),
                     "gap": gap, "gap_ci_low": gap_low, "gap_ci_high": gap_high,
                     "home_edge_beyond_model": edge, "home_edge_ci_low": edge_low, "home_edge_ci_high": edge_high})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(config.RESULTS_DIR, "covid_analysis.csv"), index=False)
    print(table.round(4).to_string(index=False))

    # figure: home edge beyond the model, per period, with 95% intervals
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(table))[::-1]
    edges = table["home_edge_beyond_model"] * 100
    err = [edges - table["home_edge_ci_low"] * 100, table["home_edge_ci_high"] * 100 - edges]
    ax.errorbar(edges, y, xerr=err, fmt="o", capsize=4)
    ax.axvline(0, linestyle="--", color="gray")
    ax.set_yticks(y)
    ax.set_yticklabels(table["period"])
    ax.set_xlabel("home shooters' edge beyond the model (percentage points, 95% interval)")
    ax.set_title("Did home teams shoot better than expected without fans?")
    fig.savefig(os.path.join(config.RESULTS_DIR, "covid_home_edge.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
