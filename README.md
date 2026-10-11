# nba-shot-calibration

**Status: in progress (setup done, models next).**

Calibrated NBA shot-make probabilities from public shot-location data, using season-based splits
that can't leak future information, a ladder of simple-to-complex models, and honest uncertainty.

## Run it
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# put the dataset zip at data/nba_data.zip (see data/README.md), then:
python -m nba_shots.clean_data
python -m nba_shots.check_data
python -m pytest
```

Data sources, terms and known issues: [DATA_CARD.md](DATA_CARD.md).
