# The baseline ladder. Every model is a function predict_xxx(train, val) that learns from the
# training shots and returns P(make) for each validation shot, so they can all be scored the same way.

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nba_shots import config
from nba_shots.features import all_features, common_action_types, distance_zone_features
from nba_shots.leakage import check_features


def predict_constant(train, val):
    """M0: every shot gets the training make rate. Everything else has to beat this."""
    return np.full(len(val), train["made"].mean())


def fit_logistic(X_train, y_train):
    """Logistic regression with the inputs rescaled first (helps it fit quickly and reliably)."""
    check_features(list(X_train.columns))
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    return model


def predict_distance_only(train, val):
    """M1: logistic regression on shot distance alone."""
    model = fit_logistic(train[["shot_distance"]], train["made"])
    return model.predict_proba(val[["shot_distance"]])[:, 1]


def predict_distance_shape_zone(train, val):
    """M2: logistic regression on distance + hinges + 3-point flag + court zone."""
    X_train = distance_zone_features(train)
    # make sure val has exactly the same columns as train (a zone missing from val becomes all 0s)
    X_val = distance_zone_features(val).reindex(columns=X_train.columns, fill_value=0)
    model = fit_logistic(X_train, train["made"])
    return model.predict_proba(X_val)[:, 1]


def predict_court_squares(train, val, size=config.BIN_SIZE_FT, k=config.SMOOTHING_K):
    """M3: the make rate of the size-by-size ft court square the shot came from.

    Each square starts with k pretend shots at the overall make rate, so a square with only a
    handful of real shots stays close to average instead of jumping to 0% or 100%.
    """
    overall = train["made"].mean()
    train = train.assign(square_x=train["loc_x"] // size, square_y=train["loc_y"] // size)
    val = val.assign(square_x=val["loc_x"] // size, square_y=val["loc_y"] // size)

    squares = train.groupby(["square_x", "square_y"])["made"].agg(["sum", "count"]).reset_index()
    squares["rate"] = (squares["sum"] + k * overall) / (squares["count"] + k)

    # a left merge keeps val's rows in their original order
    merged = val.merge(squares[["square_x", "square_y", "rate"]], on=["square_x", "square_y"], how="left")
    # a square with no training shots at all just gets the overall rate
    return merged["rate"].fillna(overall).to_numpy()


def predict_logistic_all(train, val, shot_type="family"):
    """M4: logistic regression on every feature we have (the best a straight-line model can do).

    shot_type is passed to all_features: "family" (main), "family+flags", "detailed" (raw shot
    descriptions), or "none" (no shot type, the ablation: the description is written down after the shot).
    """
    if len(train) > config.LOGISTIC_MAX_ROWS:
        train = train.sample(n=config.LOGISTIC_MAX_ROWS, random_state=0)
    common = common_action_types(train)
    X_train = all_features(train, common, shot_type)
    X_val = all_features(val, common, shot_type).reindex(columns=X_train.columns, fill_value=0)
    model = fit_logistic(X_train, train["made"])
    return model.predict_proba(X_val)[:, 1]


def predict_logistic_with_flags(train, val):
    """M4 with the shot families plus the two extra flags (comparison)."""
    return predict_logistic_all(train, val, shot_type="family+flags")


def predict_logistic_detailed(train, val):
    """M4 with the raw shot descriptions instead of families (comparison)."""
    return predict_logistic_all(train, val, shot_type="detailed")


def predict_logistic_no_shot_type(train, val):
    """M4 with no shot type at all (ablation)."""
    return predict_logistic_all(train, val, shot_type="none")


def pick_heldout_games(games, share, seed=0):
    """True for every shot from a random `share` of the games (whole games, never single shots)."""
    game_list = games.unique()
    rng = np.random.default_rng(seed)
    held_games = rng.choice(game_list, size=int(len(game_list) * share), replace=False)
    return np.asarray(games.isin(held_games))


def stopping_set(train, how=config.BOOSTING_STOPPING):
    """True for the training shots that referee "stop adding trees" (see config.BOOSTING_STOPPING)."""
    # holding out "the last season" only makes sense if there's more than one season to train on
    if how == "last_season" and train["season"].nunique() > 1:
        return np.asarray(train["season"] == train["season"].max())
    return pick_heldout_games(train["game_key"], config.BOOSTING_HELDOUT_GAMES)


def new_boosting_model(max_iter=100, warm_start=False, min_leaf=config.BOOSTING_MIN_SHOTS_PER_LEAF,
                       l2=config.BOOSTING_L2):
    """The gradient boosting model with our settings (one place, so every script uses the same ones).
    warm_start=True: each fit() keeps the trees it already has and adds more up to max_iter."""
    return HistGradientBoostingClassifier(learning_rate=config.BOOSTING_LEARNING_RATE,
                                          min_samples_leaf=min_leaf, l2_regularization=l2,
                                          max_iter=max_iter, early_stopping=False,
                                          warm_start=warm_start, random_state=0)


def fit_boosting(X, y, held, min_leaf=config.BOOSTING_MIN_SHOTS_PER_LEAF, l2=config.BOOSTING_L2):
    """Gradient boosting, adding trees in steps until it stops helping on the held-out training shots.

    held: True for the training shots used as the referee (from stopping_set). They're whole games
    or a whole season, never single shots. Validation seasons are not used here at all.
    Returns (fitted model on all training shots, number of trees chosen).
    """
    check_features(list(X.columns))
    X = X.to_numpy()  # a plain array: selecting rows from it below is cheaper than from a DataFrame
    y = np.asarray(y)

    model = new_boosting_model(warm_start=True, min_leaf=min_leaf, l2=l2)
    best_loss = np.inf
    best_trees = config.BOOSTING_TREES_STEP
    for n_trees in range(config.BOOSTING_TREES_STEP, config.BOOSTING_MAX_TREES + 1, config.BOOSTING_TREES_STEP):
        model.set_params(max_iter=n_trees)
        model.fit(X[~held], y[~held])
        loss = log_loss(y[held], model.predict_proba(X[held])[:, 1])
        print(f"    {n_trees:4d} trees: held-out log loss {loss:.5f}", flush=True)
        if loss < best_loss:
            best_loss = loss
            best_trees = n_trees
        else:
            break  # adding trees stopped helping

    # refit from scratch on ALL training shots with the chosen number of trees
    final = new_boosting_model(max_iter=best_trees, min_leaf=min_leaf, l2=l2)
    final.fit(X, y)
    return final, best_trees


def predict_boosting(train, val, shot_type="family", stopping=config.BOOSTING_STOPPING,
                     min_leaf=config.BOOSTING_MIN_SHOTS_PER_LEAF, l2=config.BOOSTING_L2,
                     with_xy=config.BOOSTING_WITH_XY, extra_cols=None):
    """M5: gradient boosting on the same features as M4 (plus, by default, the raw court coordinates).
    The other arguments default to the settings in config.py; tune_boosting.py tries other values."""
    common = common_action_types(train)
    X_train = all_features(train, common, shot_type, with_xy=with_xy, extra_cols=extra_cols)
    X_val = all_features(val, common, shot_type, with_xy=with_xy, extra_cols=extra_cols)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    held = stopping_set(train, stopping)
    model, n_trees = fit_boosting(X_train, train["made"], held, min_leaf, l2)
    print("    chosen trees:", n_trees)
    return model.predict_proba(X_val.to_numpy())[:, 1]  # array in, same as in training


def predict_boosting_with_flags(train, val):
    return predict_boosting(train, val, shot_type="family+flags")


def predict_boosting_detailed(train, val):
    return predict_boosting(train, val, shot_type="detailed")


def predict_boosting_no_shot_type(train, val):
    return predict_boosting(train, val, shot_type="none")


SKILL_COLUMNS = ["shooter_skill", "shooter_prior_shots"]
DEFENSE_COLUMNS = ["opponent_defense", "opponent_prior_shots"]


def predict_boosting_with_skill(train, val):
    """M5 + shooter skill: the main model plus the shooter's as-of skill and how many earlier shots it's based on
    (so the trees can learn to trust the skill number more when it rests on more shots)."""
    return predict_boosting(train, val, extra_cols=SKILL_COLUMNS)


def predict_boosting_with_defense(train, val):
    """M5 + opponent defense (and how many earlier shots this season it's based on)."""
    return predict_boosting(train, val, extra_cols=DEFENSE_COLUMNS)


def predict_boosting_with_both(train, val):
    """M5 + shooter skill + opponent defense."""
    return predict_boosting(train, val, extra_cols=SKILL_COLUMNS + DEFENSE_COLUMNS)
