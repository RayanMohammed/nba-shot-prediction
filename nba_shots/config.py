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
