# Split shots into train / val / test by season, and check the split is safe.

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
