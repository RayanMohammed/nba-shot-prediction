# The baseline ladder. Every model is a function predict_xxx(train, val) that learns from the
# training shots and returns P(make) for each validation shot, so they can all be scored the same way.

import numpy as np
from sklearn.linear_model import LogisticRegression
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


def predict_logistic_all(train, val, use_action_type=True):
    """M4: logistic regression on every feature we have (the best a straight-line model can do).

    use_action_type=False runs the same model without the shot description, since that column is
    written down after the shot (see leakage.FLAGGED) and we have to show results without it too.
    """
    if len(train) > config.LOGISTIC_MAX_ROWS:
        train = train.sample(n=config.LOGISTIC_MAX_ROWS, random_state=0)
    common = common_action_types(train)
    X_train = all_features(train, common, use_action_type)
    X_val = all_features(val, common, use_action_type).reindex(columns=X_train.columns, fill_value=0)
    model = fit_logistic(X_train, train["made"])
    return model.predict_proba(X_val)[:, 1]


def predict_logistic_no_action(train, val):
    """M4 without the shot description (the ablation)."""
    return predict_logistic_all(train, val, use_action_type=False)
