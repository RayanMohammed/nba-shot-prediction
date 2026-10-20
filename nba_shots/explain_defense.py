# Why did opponent defense help on validation but not on the final test?
# Hypothesis: on validation all models were compared UNcalibrated while predicting ~1 point too low, and the
# defense feature partly sensed "this season shoots hotter", i.e. it was fixing drift, not measuring defense.
# Check: calibrate every model on 2021-22, then compare on 2022-23. If defense's gain mostly disappears once
# drift is fixed by calibration, the hypothesis holds. Reads saved predictions; takes seconds.
# Run from the project folder:   python -m nba_shots.explain_defense

import os

import pandas as pd

from nba_shots import config
from nba_shots.calibrate import fit_and_apply
from nba_shots.evaluate import bootstrap_log_loss_diff, score

MAIN = "M5 boosting, families"
MODELS = [MAIN, f"M5 + shooter skill (k={config.SKILL_K})", f"M5 + opponent defense (k={config.DEFENSE_K})",
          f"M5 + both (k={config.SKILL_K}, {config.DEFENSE_K})"]


def main():
    preds = pd.read_parquet(os.path.join(config.PREDICTIONS_DIR, "skill_val.parquet"))
    fit_part = preds[preds["season"] == config.CALIBRATION_FIT_SEASON]
    check = preds[preds["season"] == config.CALIBRATION_CHECK_SEASON]

    calibrated = {}
    for name in MODELS:
        calibrated[name] = fit_and_apply(config.CALIBRATION_METHOD, fit_part[name].to_numpy(),
                                         fit_part["made"].to_numpy(), check[name].to_numpy())

    rows = []
    for name in MODELS:
        for version, p in [("uncalibrated", check[name].to_numpy()), ("calibrated", calibrated[name])]:
            p_main = check[MAIN].to_numpy() if version == "uncalibrated" else calibrated[MAIN]
            diff, low, high = bootstrap_log_loss_diff(check["made"], p, p_main, check["game_key"],
                                                      n_boot=config.N_BOOTSTRAP)
            rows.append({"model": name, "version": version, "log_loss": score(check["made"], p)["log_loss"],
                         "avg_pred": p.mean(), "actual_rate": check["made"].mean(),
                         "vs_main": diff, "vs_main_ci_low": low, "vs_main_ci_high": high})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(config.RESULTS_DIR, "explain_defense.csv"), index=False)
    print(f"On {config.CALIBRATION_CHECK_SEASON} (vs main = log loss minus the main model's, same version; negative = better):")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
