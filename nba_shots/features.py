# Turn clean shot rows into model inputs.

import pandas as pd

from nba_shots import config


def add_distance_hinges(df, knots=config.DIST_KNOTS):
    """Add a 'dist_over_k' column for each knot k: how many feet past k the shot is (0 if closer).

    A logistic regression on plain distance can only draw one straight line. These extra columns
    let it change slope at each knot, so it can follow the bends in the make-rate curve.
    """
    df = df.copy()
    for k in knots:
        df["dist_over_" + str(k)] = (df["shot_distance"] - k).clip(lower=0)
    return df


def distance_zone_features(df):
    """Inputs for M2: distance, the hinge columns, the 3-point flag, and one 0/1 column per court zone."""
    df = add_distance_hinges(df)
    hinge_cols = ["dist_over_" + str(k) for k in config.DIST_KNOTS]
    numbers = df[["shot_distance", "is_three"] + hinge_cols]
    zones = pd.get_dummies(df["basic_zone"], prefix="zone", dtype=int)
    return pd.concat([numbers, zones], axis=1)
