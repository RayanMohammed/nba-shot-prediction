# Split shots into train / val / test by season, and check the split is safe.

import os

from nba_shots import config


def assign_split(df, splits=config.SPLITS):
    """Add a 'split' column: train, val, test, test_late, or unused (season not in any split)."""
    df = df.copy()
    df["split"] = "unused"
    for name, (first, last) in splits.items():
        # season strings like "2003-04" sort in time order, so >= and <= work on them
        in_range = (df["season"] >= first) & (df["season"] <= last)
        if (df.loc[in_range, "split"] != "unused").any():
            raise ValueError("A season is in more than one split: check config.SPLITS")
        df.loc[in_range, "split"] = name
    return df


def check_splits(df, order=config.SPLIT_ORDER):
    """Raise an error if a game is in two splits, or if a later split starts before an earlier one ends."""
    splits_per_game = df.groupby("game_key")["split"].nunique()
    if (splits_per_game > 1).any():
        raise ValueError("Some games appear in more than one split")

    for i in range(len(order) - 1):
        earlier = df[df["split"] == order[i]]
        later = df[df["split"] == order[i + 1]]
        if len(earlier) == 0 or len(later) == 0:
            continue
        if earlier["game_date"].max() >= later["game_date"].min():
            raise ValueError(order[i] + " overlaps in time with " + order[i + 1])


def seasons_in(split_names, all_seasons, splits=config.SPLITS):
    """The seasons (from all_seasons) that belong to any of the named splits."""
    keep = []
    for season in all_seasons:
        for name in split_names:
            first, last = splits[name]
            if first <= season <= last:
                keep.append(season)
    return keep


def load_train_val(columns, include_old_seasons=True):
    """Load ONLY the train and val seasons (covid and test are never read).

    include_old_seasons=False skips training seasons before config.MAIN_TRAIN_START (saves memory
    when the all-seasons comparison isn't needed); the second return value is then None.
    Returns (main training shots from config.MAIN_TRAIN_START on, all training shots, validation shots).
    """
    from nba_shots.check_data import load_clean_data  # imported here to avoid a circular import

    all_seasons = [f.replace(".parquet", "") for f in sorted(os.listdir(config.CLEAN_DIR))]
    seasons = seasons_in(["train", "val"], all_seasons)
    if not include_old_seasons:
        seasons = [s for s in seasons if s >= config.MAIN_TRAIN_START]
    df = assign_split(load_clean_data(config.CLEAN_DIR, columns=columns, seasons=seasons))
    check_splits(df)
    all_train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    del df  # all_train and val are copies, so free the full table to save memory
    train = all_train[all_train["season"] >= config.MAIN_TRAIN_START]
    print("main train shots:", len(train), "| all-season train shots:", len(all_train), "| val shots:", len(val))
    if not include_old_seasons:
        return all_train, None, val  # all_train already only has the main seasons
    return train, all_train, val
