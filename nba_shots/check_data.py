# Step 2: sanity-check the clean data. Stops with an error if something is seriously wrong.
# Run from the project folder:   python -m nba_shots.check_data

import os

import numpy as np
import pandas as pd

from nba_shots import config

# only load the columns the checks need (saves a lot of memory)
COLUMNS = ["season", "game_key", "game_date", "home_team", "player_id", "quarter",
           "loc_x", "loc_y", "shot_distance", "is_three", "made"]


def load_clean_data(clean_dir, columns=None, seasons=None):
    """Read the clean season files and stack them into one DataFrame.

    columns: only load these columns (None = all of them).
    seasons: only load these seasons, e.g. ["2021-22", "2022-23"] (None = all of them).
    """
    parts = []
    for file_name in sorted(os.listdir(clean_dir)):
        if not file_name.endswith(".parquet"):
            continue
        season = file_name.replace(".parquet", "")
        if seasons is not None and season not in seasons:
            continue
        df = pd.read_parquet(os.path.join(clean_dir, file_name), columns=columns)
        # store text columns compactly: normal pandas text uses about 3x the memory
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype("string[pyarrow]")
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def run_checks(df):
    """Count possible problems. Returns (dict of counts, list of serious problems)."""
    counts = {}
    counts["rows"] = len(df)
    counts["missing_values"] = int(df.isna().sum().sum())
    counts["labels_not_0_or_1"] = int((~df["made"].isin([0, 1])).sum())

    off_court = (df["loc_x"].abs() > 25.5) | (df["loc_y"] < 0) | (df["loc_y"] > 94)
    counts["shots_off_the_court"] = int(off_court.sum())

    # distance from the hoop using x and y, compared with the distance column
    dist_from_xy = np.sqrt(df["loc_x"] ** 2 + (df["loc_y"] - config.HOOP_Y) ** 2)
    mismatch = (dist_from_xy - df["shot_distance"]).abs() > 1.5
    counts["share_xy_disagrees_with_distance"] = float(mismatch.mean())

    # each game should have exactly one date and one home team
    per_game = df.groupby("game_key")[["game_date", "home_team"]].nunique()
    counts["games_with_conflicting_info"] = int((per_game > 1).any(axis=1).sum())

    # small oddities we just keep an eye on
    counts["threes_under_21ft"] = int(((df["is_three"] == 1) & (df["shot_distance"] < 21)).sum())
    counts["twos_over_24ft"] = int(((df["is_three"] == 0) & (df["shot_distance"] > 24)).sum())
    counts["make_rate"] = float(df["made"].mean())

    problems = []
    if counts["missing_values"] > 0:
        problems.append("missing values")
    if counts["labels_not_0_or_1"] > 0:
        problems.append("labels that aren't 0 or 1")
    if counts["shots_off_the_court"] > 0:
        problems.append("shots off the court")
    if counts["share_xy_disagrees_with_distance"] > 0.001:
        problems.append("x/y don't match shot distance (coordinate scale problem?)")
    if counts["games_with_conflicting_info"] > 0:
        problems.append("a game_key with more than one date or home team")
    return counts, problems


def main():
    df = load_clean_data(config.CLEAN_DIR, columns=COLUMNS)
    counts, problems = run_checks(df)

    for name, value in counts.items():
        print(name, ":", value)
    pd.Series(counts).to_csv(os.path.join(config.RESULTS_DIR, "data_checks.csv"), header=False)

    by_season = df.groupby("season").agg(shots=("made", "size"), make_rate=("made", "mean"),
                                         games=("game_key", "nunique"))
    by_season.to_csv(os.path.join(config.RESULTS_DIR, "shots_by_season.csv"))

    if len(problems) > 0:
        raise ValueError("Data problems found: " + "; ".join(problems))
    print("All checks passed.")


if __name__ == "__main__":
    main()
