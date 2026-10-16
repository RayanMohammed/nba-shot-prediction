# Turn clean shot rows into model inputs.

import numpy as np
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


def add_angle(df):
    """Angle of the shot from straight-on, in degrees: 0 = straight at the hoop, 90 = along the
    baseline, over 90 = from behind the backboard. Left and right sides count the same."""
    df = df.copy()
    sideways = df["loc_x"].abs()
    forward = df["loc_y"] - config.HOOP_Y
    df["angle"] = np.degrees(np.arctan2(sideways, forward))
    return df


def add_game_clock(df):
    """Seconds left in the period, an end-of-period flag, and the period (overtimes grouped as 5)."""
    df = df.copy()
    df["secs_left_in_period"] = df["mins_left"] * 60 + df["secs_left"]
    df["last_5_secs"] = (df["secs_left_in_period"] <= 5).astype(int)
    df["period"] = df["quarter"].clip(upper=5)
    return df


def add_is_home(df):
    """1 if the shooter's team is the home team.

    The data gives the shooter's team as a NAME ("Boston Celtics") but home/away as ABBREVIATIONS
    ("BOS"), so first find each team's abbreviation: it's the one that shows up in every one of that
    team's games, while each opponent only shows up in a few. Done per season because teams moved
    and renamed (Seattle -> OKC, New Jersey -> Brooklyn).
    """
    games = df[["season", "team_name", "game_key", "home_team", "away_team"]].drop_duplicates()
    home = games[["season", "team_name", "home_team"]].rename(columns={"home_team": "abbr"})
    away = games[["season", "team_name", "away_team"]].rename(columns={"away_team": "abbr"})
    counts = pd.concat([home, away]).groupby(["season", "team_name", "abbr"]).size().reset_index(name="games")

    # for each team-season keep the abbreviation that appears in the most games
    counts = counts.sort_values("games", ascending=False)
    own = counts.drop_duplicates(subset=["season", "team_name"])[["season", "team_name", "abbr"]]

    df = df.merge(own, on=["season", "team_name"], how="left")
    df["is_home"] = (df["abbr"] == df["home_team"]).astype(int)
    return df.drop(columns=["abbr"])


def common_action_types(train, min_shots=config.MIN_ACTION_SHOTS):
    """Shot descriptions with at least min_shots TRAINING shots. Everything else becomes "Other"."""
    counts = train["action_type"].value_counts()
    return list(counts[counts >= min_shots].index)


def shot_family(action_type):
    """Group one shot description into a family, e.g. "Driving Floating Bank Jump Shot" -> "floater"."""
    for family, keywords in config.SHOT_FAMILIES:
        for word in keywords:
            if word in action_type:
                return family
    return "jumper"


def all_features(df, common_actions, shot_type="family"):
    """Inputs for M4: everything from M2, plus angle, side of the court, game clock, home/away,
    and the kind of shot.

    shot_type: "family"   -> the ~9 shot families (main model, and what the app will offer)
               "detailed" -> the raw shot descriptions (comparison)
               "none"     -> no shot type at all (ablation)
    """
    df = add_distance_hinges(df)
    df = add_angle(df)
    df = add_game_clock(df)
    df = add_is_home(df)

    hinge_cols = ["dist_over_" + str(k) for k in config.DIST_KNOTS]
    number_cols = ["shot_distance", "is_three", "angle", "secs_left_in_period", "last_5_secs",
                   "is_home"] + hinge_cols
    parts = [
        df[number_cols],
        pd.get_dummies(df["basic_zone"], prefix="zone", dtype=int),
        pd.get_dummies(df["zone_name"], prefix="side", dtype=int),
        pd.get_dummies(df["period"], prefix="period", dtype=int),
    ]
    if shot_type == "family":
        # .unique() first, so shot_family runs once per distinct description, not once per shot
        families = {a: shot_family(a) for a in df["action_type"].unique()}
        parts.append(pd.get_dummies(df["action_type"].map(families), prefix="family", dtype=int))
    elif shot_type == "detailed":
        action = df["action_type"].where(df["action_type"].isin(common_actions), "Other")
        parts.append(pd.get_dummies(action, prefix="action", dtype=int))
    # 32-bit decimals use half the memory of pandas' default 64-bit numbers, plenty precise for this
    return pd.concat(parts, axis=1).astype("float32")
