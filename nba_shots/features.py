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


def add_team_abbr(df):
    """Add team_abbr: the shooter's team as an abbreviation ("BOS").

    The data gives the shooter's team as a NAME ("Boston Celtics") but home/away as ABBREVIATIONS,
    so find each team's abbreviation: it's the one that shows up in every one of that team's games,
    while each opponent only shows up in a few. Done per season because teams moved and renamed
    (Seattle -> OKC, New Jersey -> Brooklyn).
    """
    games = df[["season", "team_name", "game_key", "home_team", "away_team"]].drop_duplicates()
    home = games[["season", "team_name", "home_team"]].rename(columns={"home_team": "team_abbr"})
    away = games[["season", "team_name", "away_team"]].rename(columns={"away_team": "team_abbr"})
    counts = pd.concat([home, away]).groupby(["season", "team_name", "team_abbr"]).size().reset_index(name="games")

    # for each team-season keep the abbreviation that appears in the most games
    counts = counts.sort_values("games", ascending=False)
    own = counts.drop_duplicates(subset=["season", "team_name"])[["season", "team_name", "team_abbr"]]
    return df.merge(own, on=["season", "team_name"], how="left")


def add_is_home(df):
    """1 if the shooter's team is the home team."""
    df = add_team_abbr(df)
    df["is_home"] = (df["team_abbr"] == df["home_team"]).astype(int)
    return df.drop(columns=["team_abbr"])


def add_opponent(df):
    """Add opponent: the DEFENDING team's abbreviation (whichever of home/away isn't the shooter's team)."""
    df = add_team_abbr(df)
    df["opponent"] = np.where(df["team_abbr"] == df["home_team"], df["away_team"], df["home_team"])
    return df.drop(columns=["team_abbr"])


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


def all_features(df, common_actions, shot_type="family", with_xy=False, extra_cols=None):
    """Inputs for M4: everything from M2, plus angle, side of the court, game clock, home/away,
    and the kind of shot.

    shot_type: "family"       -> the 9 shot families (main model, and what the app will offer)
               "family+flags" -> families + on_the_move + alley_oop (comparison: the flags drift over seasons)
               "detailed"     -> the raw shot descriptions (comparison)
               "none"         -> no shot type at all (ablation)
    with_xy: also include the raw court coordinates (useful for tree models, which can split on them)
    extra_cols: more number columns already in df to include as-is (e.g. the shooter skill columns)
    """
    df = add_distance_hinges(df)
    df = add_angle(df)
    df = add_game_clock(df)
    df = add_is_home(df)

    hinge_cols = ["dist_over_" + str(k) for k in config.DIST_KNOTS]
    number_cols = ["shot_distance", "is_three", "angle", "secs_left_in_period", "last_5_secs",
                   "is_home"] + hinge_cols
    if with_xy:
        number_cols = number_cols + ["loc_x", "loc_y"]
    if extra_cols is not None:
        number_cols = number_cols + extra_cols
    parts = [
        df[number_cols],
        pd.get_dummies(df["basic_zone"], prefix="zone", dtype=int),
        pd.get_dummies(df["zone_name"], prefix="side", dtype=int),
        pd.get_dummies(df["period"], prefix="period", dtype=int),
    ]
    if shot_type in ["family", "family+flags"]:
        # .unique() first, so shot_family runs once per distinct description, not once per shot
        families = {a: shot_family(a) for a in df["action_type"].unique()}
        parts.append(pd.get_dummies(df["action_type"].map(families), prefix="family", dtype=int))
    if shot_type == "family+flags":
        # "Driving|Running|Cutting" means "contains any of these words"
        moving = df["action_type"].str.contains("|".join(config.MOVING_WORDS))
        oop = df["action_type"].str.contains("Alley Oop")
        parts.append(pd.DataFrame({"on_the_move": moving.astype(int), "alley_oop": oop.astype(int)}))
    if shot_type == "detailed":
        action = df["action_type"].where(df["action_type"].isin(common_actions), "Other")
        parts.append(pd.get_dummies(action, prefix="action", dtype=int))
    # 32-bit decimals use half the memory of pandas' default 64-bit numbers, plenty precise for this
    return pd.concat(parts, axis=1).astype("float32")


def skill_category(df):
    """Shot category for shooter skill: "rim" (restricted area), "three", or "two" (everything else)."""
    return np.where(df["basic_zone"] == "Restricted Area", "rim",
                    np.where(df["is_three"] == 1, "three", "two"))


def expected_make_rates(train):
    """League make rate for each (shot family, court zone), from TRAINING shots only.
    This is what an average player makes on that kind of shot."""
    families = {a: shot_family(a) for a in train["action_type"].unique()}
    table = pd.DataFrame({"family": train["action_type"].map(families).astype(str).to_numpy(),
                          "basic_zone": train["basic_zone"].astype(str).to_numpy(),
                          "made": train["made"].to_numpy()})
    return table.groupby(["family", "basic_zone"])["made"].mean().reset_index(name="expected")


def add_as_of_history(df, expected, who, name, by_season=False):
    """For each shot, the record of `who` (a shooter, or a defending team) in EARLIER GAMES ONLY,
    for this category of shot (rim / two / three):
         <name>_prior_shots: how many such shots
         <name>_prior_resid: total (made - expected) over them, i.e. makes above an average player
    The shot's own game and every later game are left out, so nothing from the future leaks in.
    by_season=True restarts the count every season.
    df must contain every season you want counted as history (earlier seasons included).
    """
    out = df.copy()
    families = {a: shot_family(a) for a in out["action_type"].unique()}
    keys_for_expected = pd.DataFrame({"family": out["action_type"].map(families).astype(str).to_numpy(),
                                      "basic_zone": out["basic_zone"].astype(str).to_numpy()})
    looked_up = keys_for_expected.merge(expected, on=["family", "basic_zone"], how="left")["expected"]
    # a (family, zone) combination that never happened in training gets the average expected rate
    out["expected"] = looked_up.fillna(expected["expected"].mean()).to_numpy()
    out["resid"] = out["made"] - out["expected"]
    out["skill_cat"] = skill_category(out)

    # one row per (who, shot category, [season,] game), in date order
    group = [who, "skill_cat"] + (["season"] if by_season else [])
    keys = group + ["game_date", "game_key"]
    per_game = out.groupby(keys)["resid"].agg(["sum", "count"]).reset_index().sort_values(keys)
    by_who = per_game.groupby(group)
    # running total INCLUDING this game, minus this game = total of earlier games only
    per_game[name + "_prior_resid"] = by_who["sum"].cumsum() - per_game["sum"]
    per_game[name + "_prior_shots"] = by_who["count"].cumsum() - per_game["count"]

    out = out.merge(per_game[keys + [name + "_prior_resid", name + "_prior_shots"]], on=keys, how="left")
    return out.drop(columns=["expected", "resid", "skill_cat"])


def add_shooter_history(df, expected):
    """The shooter's record in earlier games, across seasons (players carry their skill with them)."""
    return add_as_of_history(df, expected, who="player_id", name="shooter")


def add_defense_history(df, expected):
    """The DEFENDING team's record in earlier games of the SAME season (rosters change every summer):
    makes it allowed above what an average defense allows on those shots. Positive = leaky defense."""
    df = add_opponent(df)
    return add_as_of_history(df, expected, who="opponent", name="opponent", by_season=True)


def add_shooter_skill(df, k=config.SKILL_K):
    """shooter_skill = makes above average in earlier games / (earlier shots + k).
    The + k pulls players with few shots toward 0 (average). Needs add_shooter_history first."""
    df = df.copy()
    df["shooter_skill"] = df["shooter_prior_resid"] / (df["shooter_prior_shots"] + k)
    return df


def add_opponent_defense(df, k=config.DEFENSE_K):
    """opponent_defense = makes allowed above average earlier this season / (earlier shots + k).
    Positive = the defense has been leaky. Needs add_defense_history first."""
    df = df.copy()
    df["opponent_defense"] = df["opponent_prior_resid"] / (df["opponent_prior_shots"] + k)
    return df
