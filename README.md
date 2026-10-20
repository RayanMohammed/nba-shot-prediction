# nba-shot-calibration

**Status: in progress (data pipeline, model ladder, boosting and recalibration code done).**

Calibrated NBA shot-make probabilities from public shot-location data, using season-based splits
that can't leak future information, a ladder of simple-to-complex models, and honest uncertainty.

## Run it
Needs Python 3.10 to 3.13 (the pinned packages don't have ready-made builds for 3.14 yet).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# put the dataset zip at data/nba_data.zip (see data/README.md), then:
python -m nba_shots.clean_data
python -m nba_shots.check_data
python -m nba_shots.train_baselines   # M0-M4, about 1 minute
python -m nba_shots.tune_boosting     # compares boosting setups, ~10 minutes
python -m nba_shots.train_boosting    # M5, several minutes
python -m nba_shots.calibrate         # M7 recalibration, seconds
python -m nba_shots.compare_models    # confidence intervals, under a minute
python -m nba_shots.check_overfitting # training vs unseen shots, several minutes
python -m pytest
```

Data sources, terms and known issues: [DATA_CARD.md](DATA_CARD.md).
