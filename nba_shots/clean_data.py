# Step 1: turn the raw season CSVs into clean files we can model with.
# Run from the project folder:   python -m nba_shots.clean_data

import hashlib
import os
import zipfile

import pandas as pd

from nba_shots import config


def check_zip(zip_path, expected_hash):
    """Make sure the zip is the exact same download the project was built with."""
    with open(zip_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("This zip is a different version of the dataset than the project expects.")


def unzip_if_needed(zip_path, raw_dir):
    """Unzip the season CSVs into raw_dir (skipped if they're already there)."""
    os.makedirs(raw_dir, exist_ok=True)
    if len(os.listdir(raw_dir)) == 0:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(raw_dir)


def clean_season(raw):
    """Clean one season of raw shots. Returns (clean DataFrame, dict of how many rows were dropped and why)."""
    df = raw.copy()

    # drop columns that repeat other columns
    df = df.drop(columns=["SEASON_1", "EVENT_TYPE", "ZONE_ABB"])

    # nicer column names
    df = df.rename(columns={"SEASON_2": "season", "SHOT_MADE": "made"})
    df.columns = df.columns.str.lower()

    # types
    df["game_date"] = pd.to_datetime(df["game_date"], format="%m-%d-%Y")
    df["made"] = df["made"].astype(int)
    df["is_three"] = (df["shot_type"] == "3PT Field Goal").astype(int)

    # fix the seasons whose coordinates were saved at 1/10 scale
    season = df["season"].iloc[0]
    if season in config.SCALED_SEASONS:
        df["loc_x"] = df["loc_x"] * 10
        df["loc_y"] = (df["loc_y"] - config.HOOP_Y) * 10

    # GAME_ID alone isn't unique: five 2020-21 opening games all have the placeholder 22000000.
    # Adding the date and home team makes every game's key unique.
    df["game_key"] = (df["game_id"] + "_"
                      + df["game_date"].dt.strftime("%Y-%m-%d") + "_"
                      + df["home_team"])

    # drop rows that are exact copies of another row
    rows_before = len(df)
    df = df.drop_duplicates()
    duplicates_dropped = rows_before - len(df)

    # drop backcourt heaves: end-of-quarter desperation shots (about 2% go in).
    # This model is for normal shot attempts, and heaves would give it easy, free credit.
    is_heave = df["basic_zone"] == "Backcourt"
    heaves_dropped = int(is_heave.sum())
    df = df[~is_heave]

    dropped = {"duplicates_dropped": duplicates_dropped, "heaves_dropped": heaves_dropped}
    return df, dropped


def main():
    check_zip(config.DATA_ZIP, config.ZIP_SHA256)
    unzip_if_needed(config.DATA_ZIP, config.RAW_DIR)
    os.makedirs(config.CLEAN_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    log = []
    for file_name in sorted(os.listdir(config.RAW_DIR)):
        if not file_name.endswith(".csv"):
            continue
        # read IDs as text so pandas doesn't turn them into numbers
        raw = pd.read_csv(os.path.join(config.RAW_DIR, file_name),
                          dtype={"GAME_ID": str, "PLAYER_ID": str, "TEAM_ID": str})
        clean, dropped = clean_season(raw)
        season = clean["season"].iloc[0]
        clean.to_parquet(os.path.join(config.CLEAN_DIR, season + ".parquet"), index=False)

        print(season, "rows:", len(clean), "duplicates dropped:", dropped["duplicates_dropped"],
              "heaves dropped:", dropped["heaves_dropped"])
        log.append({"season": season, "rows": len(clean),
                    "duplicates_dropped": dropped["duplicates_dropped"],
                    "heaves_dropped": dropped["heaves_dropped"]})

    pd.DataFrame(log).to_csv(os.path.join(config.RESULTS_DIR, "clean_data_log.csv"), index=False)


if __name__ == "__main__":
    main()
