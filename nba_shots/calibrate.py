# M7: recalibrate the main model. No training of the main model happens here: it reads the validation
# predictions saved by train_boosting, learns a small correction on the first validation season,
# and checks that correction on the second one. Then it runs overfitting checks on the corrections.
# Takes under a minute.
# Run from the project folder:   python -m nba_shots.calibrate

import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from nba_shots import config
from nba_shots.evaluate import (bootstrap_log_loss_diff, calibration_table, full_report, logit,
                                plot_calibration)


def fit_platt(p, y):
    """Platt scaling: a 1-feature logistic regression on logit(p).
    It can only shift predictions up/down and stretch/squeeze them, so it can't overfit much."""
    model = LogisticRegression(C=1e6)
    model.fit(logit(np.asarray(p)).reshape(-1, 1), y)
    return model


def apply_platt(model, p):
    return model.predict_proba(logit(np.asarray(p)).reshape(-1, 1))[:, 1]


def hinge_features(p):
    """logit(p), plus one hinge column per knot: how far past the knot the prediction is (0 if below).
    Same idea as the distance hinges in features.py, on the log-odds scale."""
    z = logit(np.asarray(p))
    columns = [z]
    for knot in config.CALIBRATION_KNOTS:
        columns.append(np.clip(z - logit(knot), 0, None))
    return np.column_stack(columns)


def fit_smooth(p, y):
    """Platt + hinges: a smooth correction curve that can bend at a few points. Only
    1 + len(CALIBRATION_KNOTS) slopes and an intercept to learn, so very little room to memorize."""
    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(hinge_features(p), y)
    return model


def apply_smooth(model, p):
    return model.predict_proba(hinge_features(p))[:, 1]


def fit_isotonic(p, y):
    """Isotonic regression: any curve that only goes up (higher prediction -> higher corrected value).
    More flexible than Platt, so it can fix odd-shaped errors but can also chase noise."""
    model = IsotonicRegression(out_of_bounds="clip")  # predictions outside the fitted range use the edge value
    model.fit(p, y)
    return model


def fit_calibrator(method, p, y):
    """Learn a correction of the given kind ("Platt", "smooth" or "isotonic")."""
    if method == "Platt":
        return fit_platt(p, y)
    if method == "smooth":
        return fit_smooth(p, y)
    return fit_isotonic(p, y)


def apply_calibrator(method, calibrator, p):
    """Apply a correction learned by fit_calibrator to new predictions."""
    if method == "Platt":
        return apply_platt(calibrator, p)
    if method == "smooth":
        return apply_smooth(calibrator, p)
    return calibrator.predict(p)


def fit_and_apply(method, p_fit, y_fit, p_new):
    """Learn a correction on (p_fit, y_fit), apply it to p_new."""
    return apply_calibrator(method, fit_calibrator(method, p_fit, y_fit), p_new)


def overfitting_checks(preds):
    """Is the correction learning real patterns, or memorizing the season it learned from?

    1. In-sample vs out-of-sample: an overfit correction looks great on the shots it learned from
       and much worse on new shots. A real one helps about equally on both.
    2. Both directions: learn on 2021-22 and check on 2022-23, then the other way around.
    3. Game-level bootstrap: is the out-of-sample improvement bigger than luck?
    4. Learn from a small random 10% of the season: if the gain mostly survives, the correction
       isn't depending on fine details of one particular set of shots.
    """
    seasons = [config.CALIBRATION_FIT_SEASON, config.CALIBRATION_CHECK_SEASON]
    rows = []
    for fit_season, check_season in [(seasons[0], seasons[1]), (seasons[1], seasons[0])]:
        fit_part = preds[preds["season"] == fit_season]
        check_part = preds[preds["season"] == check_season]
        p_fit, y_fit = fit_part[config.MAIN_MODEL].to_numpy(), fit_part["made"].to_numpy()
        p_check, y_check = check_part[config.MAIN_MODEL].to_numpy(), check_part["made"].to_numpy()

        # a random 10% of the fit season's shots, for check 4
        small = np.random.default_rng(0).uniform(0, 1, len(p_fit)) < 0.1

        for method in ["Platt", "smooth", "isotonic"]:
            gain_in = log_loss(y_fit, p_fit) - log_loss(y_fit, fit_and_apply(method, p_fit, y_fit, p_fit))
            corrected = fit_and_apply(method, p_fit, y_fit, p_check)
            gain_out = log_loss(y_check, p_check) - log_loss(y_check, corrected)
            diff, low, high = bootstrap_log_loss_diff(y_check, corrected, p_check, check_part["game_key"],
                                                      n_boot=config.N_BOOTSTRAP)
            small_corrected = fit_and_apply(method, p_fit[small], y_fit[small], p_check)
            gain_small = log_loss(y_check, p_check) - log_loss(y_check, small_corrected)
            print(f"{method:9s} learn {fit_season} -> check {check_season}: gain on its own shots {gain_in:.4f} | "
                  f"on new season {gain_out:.4f} [CI {-high:.4f} to {-low:.4f}] | learned from 10%: {gain_small:.4f}")
            rows.append({"method": method, "learn_on": fit_season, "check_on": check_season,
                         "gain_in_sample": gain_in, "gain_out_of_sample": gain_out,
                         "gain_ci_low": -high, "gain_ci_high": -low, "gain_learned_from_10pct": gain_small})

        # how coarse is the isotonic staircase? (few big steps = little room to memorize)
        steps = pd.Series(fit_isotonic(p_fit, y_fit).predict(p_fit)).value_counts()
        print(f"isotonic learned on {fit_season}: {len(steps)} steps, smallest step holds {steps.min()} shots, "
              f"median step {int(steps.median())} shots")
    return pd.DataFrame(rows)


def main():
    preds = pd.read_parquet(os.path.join(config.PREDICTIONS_DIR, "boosting_val.parquet"))
    fit_part = preds[preds["season"] == config.CALIBRATION_FIT_SEASON]
    check_part = preds[preds["season"] == config.CALIBRATION_CHECK_SEASON]
    p_fit = fit_part[config.MAIN_MODEL].to_numpy()
    p_check = check_part[config.MAIN_MODEL].to_numpy()
    print("calibrators learn on", config.CALIBRATION_FIT_SEASON, len(fit_part), "shots | checked on",
          config.CALIBRATION_CHECK_SEASON, len(check_part), "shots")

    versions = {"M5 uncalibrated": p_check}
    for method in ["Platt", "smooth", "isotonic"]:
        versions["M7 " + method] = fit_and_apply(method, p_fit, fit_part["made"], p_check)

    rows = []
    tables = {}
    for name, p in versions.items():
        r = full_report(check_part["made"], p)
        print(f"{name:18s} log loss {r['log_loss']:.4f} | slope {r['calib_slope']:.3f} | "
              f"intercept {r['calib_intercept']:+.3f} | ECE {r['ece']:.4f} | "
              f"avg pred {r['avg_pred']:.4f} vs actual {r['actual_rate']:.4f}")
        r["model"] = name
        rows.append(r)
        tables[name] = calibration_table(check_part["made"], p)

    pd.DataFrame(rows).to_csv(os.path.join(config.RESULTS_DIR, "calibration_check.csv"), index=False)
    plot_calibration(tables, os.path.join(config.RESULTS_DIR, "calibration_m7_check.png"),
                     "Recalibration, checked on " + config.CALIBRATION_CHECK_SEASON)

    print("\nOverfitting checks for the corrections (gain = how much log loss went DOWN; bigger is better)")
    checks = overfitting_checks(preds)
    checks.to_csv(os.path.join(config.RESULTS_DIR, "calibration_overfit_checks.csv"), index=False)


if __name__ == "__main__":
    main()
