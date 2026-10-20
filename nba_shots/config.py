# All the settings the project uses, in one place.
# If a decision changes (like which seasons are the test set), it changes here and nowhere else.
# Paths assume you run commands from the main project folder.

DATA_ZIP = "data/nba_data.zip"
RAW_DIR = "data/raw"
CLEAN_DIR = "data/clean"
RESULTS_DIR = "results"

# Fingerprint of the exact zip this project was built with (21 seasons, 2003-04 to 2023-24).
# If someone downloads a different version, clean_data.py will stop and say so.
ZIP_SHA256 = "edf81d754a6bda8fa3f67c140b4ea4cd45ada8a47569809f28b1f8255eb65d62"

# The hoop is at x = 0, y = 5.25 ft from the baseline.
HOOP_Y = 5.25

# These seasons have LOC_X / LOC_Y saved at 1/10 scale (found while exploring the data).
# Fix:  x = 10 * LOC_X   and   y = 10 * (LOC_Y - 5.25)
SCALED_SEASONS = ["2019-20", "2020-21", "2021-22"]

# Split, decided Oct 11, 2026. Locked: don't change this after looking at any test results.
# Whole seasons only, so a game is never in two splits and training always comes before testing.
SPLITS = {
    "train": ("2003-04", "2018-19"),
    "covid": ("2019-20", "2020-21"),  # set aside, not trained on: used later to study fans vs no fans
    "val": ("2021-22", "2022-23"),    # comparing models, tuning, calibration
    "test": ("2023-24", "2023-24"),   # final grade, used once at the very end
}
SPLIT_ORDER = ["train", "covid", "val", "test"]

# Main setup: train only on 2012-13 onward (decided Oct 15, 2026, from validation results).
# Older seasons use different shot labels and a different style of play. Training on ALL train
# seasons (2003-04 on) is kept as the comparison.
MAIN_TRAIN_START = "2012-13"

# --- modeling settings ---

# Distances (ft) where the make-rate curve bends, read off the TRAINING seasons only:
# steep drop from 0 to 3 ft, nearly flat from 3 to 23 ft, slow decline past 23 ft.
DIST_KNOTS = [1, 3, 23]

# Court-square baseline (M3): size of each square, and how many "average" shots
# every square starts with, so squares with only a few shots don't get extreme rates.
BIN_SIZE_FT = 2
SMOOTHING_K = 20

# M4 (logistic regression, many features): fit on a random sample of this many training shots.
# It only learns ~60 numbers, so 1M shots is plenty, and the full 3.2M needs several GB of memory.
LOGISTIC_MAX_ROWS = 1_000_000

# Shot descriptions (ACTION_TYPE) with fewer training shots than this are grouped into "Other"
# (only used by the "detailed" comparison; the main model uses shot families, see features.py)
MIN_ACTION_SHOTS = 2000

# Shot families: ACTION_TYPE grouped by keyword. Checked in this order, first match wins,
# anything left over is a "jumper". "Bank" is ignored on purpose: bank-labeled shots go in far more
# often than their non-bank versions (57% vs 34% for jump shots), probably because a shot gets the
# label mostly when it banks IN. Checked Oct 15: dropping "Bank" barely changes validation scores.
SHOT_FAMILIES = [
    ("dunk", ["Dunk"]),
    ("tip/putback", ["Tip", "Putback"]),
    ("layup", ["Layup", "Finger Roll"]),
    ("hook", ["Hook"]),
    ("floater", ["Floating"]),
    ("step-back", ["Step Back"]),
    ("pull-up", ["Pullup", "Pull-Up"]),
    ("turnaround/fadeaway", ["Turnaround", "Fadeaway"]),
]

# Extra shot details, only used in the "family+flags" comparison. Not in the main model: checked Oct 15,
# "Driving" went from 30% to 53% of layups between 2012-13 and 2022-23 and its effect flipped sign.
# on_the_move = the description contains one of these words; alley_oop = it contains "Alley Oop"
MOVING_WORDS = ["Driving", "Running", "Cutting"]

# M5 gradient boosting (scikit-learn's HistGradientBoostingClassifier)
BOOSTING_LEARNING_RATE = 0.1
BOOSTING_MIN_SHOTS_PER_LEAF = 1000  # every leaf must hold at least this many shots (was 100; tuned Oct 19)
BOOSTING_TREES_STEP = 50            # add trees 50 at a time while checking the held-out games
BOOSTING_MAX_TREES = 1000
BOOSTING_HELDOUT_GAMES = 0.1        # share of TRAINING games held out to decide when to stop adding trees

# Which training shots referee "stop adding trees":
#   "random_games" -> a random 10% of training games (same seasons as the rest of training)
#   "last_season"  -> every game of the latest training season (closer to predicting the future)
BOOSTING_STOPPING = "last_season"   # chosen Oct 19 (tune_boosting.py); was "random_games"
BOOSTING_L2 = 1.0                   # penalty on extreme leaf values (0 = none); was 0, tuned Oct 19
BOOSTING_WITH_XY = False            # raw court coordinates: off since Oct 19. They helped on 2018-19 but not
                                    # at all on 2021-23, i.e. they taught era-specific detail

# Validation predictions saved by the training scripts (inside data/, so never uploaded)
PREDICTIONS_DIR = "data/predictions"

# The model everything else is built around (decided Oct 15, 2026)
MAIN_MODEL = "M5 boosting, families"

# M7 recalibration: learn the correction on the first validation season, check it on the second,
# so the correction is never graded on the shots it learned from
CALIBRATION_FIT_SEASON = "2021-22"
CALIBRATION_CHECK_SEASON = "2022-23"

# Bootstrap confidence intervals: how many times to resample the games
N_BOOTSTRAP = 1000

# check_overfitting.py: grow the boosting model this far (past where it normally stops) to see the full curve
OVERFIT_CHECK_MAX_TREES = 600

# Smooth calibrator (Platt + hinges): predicted probabilities where the correction curve may bend
CALIBRATION_KNOTS = [0.35, 0.45, 0.55, 0.65]

# The calibrator the final model uses (chosen Oct 19): "smooth" kept ~90% of its gain when learned from
# only 10% of a season; isotonic's gain collapsed to ~0 there (calibration_overfit_checks.csv)
CALIBRATION_METHOD = "smooth"

# The final model + calibrator, saved by final_test.py and loaded by the app (inside data/, never uploaded)
FINAL_MODEL_PATH = "data/final_model.pkl"

# Shooter skill: shrinkage = every player starts with this many pretend league-average shots per shot
# category, so a player with few shots stays near average. Several values are tried on validation.
SKILL_K_OPTIONS = [100, 300, 1000]
SKILL_K = 1000  # chosen Oct 19 (train_skill.py); 100/300/1000 all scored within 0.0001, so k barely matters

# Opponent defense: same idea, per defending team, restarting each season. A team faces ~85 shots a game,
# so its record fills up fast; heavier shrinkage values are tried.
DEFENSE_K_OPTIONS = [300, 1000, 3000]
DEFENSE_K = 1000  # chosen Oct 19 (train_skill.py); 300/1000/3000 all scored within 0.0001

# covid_analysis.py: periods with and without fans (approximate: attendance varied by arena in 2020-21,
# and this data has no attendance column). (label, first date, last date), dates inclusive.
COVID_PERIODS = [
    ("2019-20 before shutdown (fans)", "2019-10-01", "2020-03-11"),
    ("2019-20 bubble (no fans, neutral site)", "2020-07-30", "2020-08-31"),
    ("2020-21 Dec-Feb (mostly no fans)", "2020-12-01", "2021-02-28"),
    ("2020-21 Mar-May (some limited fans)", "2021-03-01", "2021-05-31"),
    ("2021-22 (fans back)", "2021-10-01", "2022-06-30"),
]
