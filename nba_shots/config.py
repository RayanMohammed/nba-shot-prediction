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

# PROPOSED split, not frozen yet. Whole seasons only, so a game can never be in two splits
# and everything we train on happened before everything we test on.
SPLITS = {
    "train": ("2003-04", "2020-21"),
    "val": ("2021-22", "2021-22"),        # for tuning and calibration
    "test": ("2022-23", "2022-23"),       # used once, at the very end
    "test_late": ("2023-24", "2023-24"),  # a second, later test to see drift
}
SPLIT_ORDER = ["train", "val", "test", "test_late"]
