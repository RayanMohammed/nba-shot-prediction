# Data card

## Source
- **Dataset:** "NBA Shots", regular-season field-goal attempts 2003-04 through 2023-24, one CSV per season
  (21 files, 4,231,262 rows). Compiled by DomSamangy from NBA.com; mirrored on Kaggle (mexwell/nba-shots).
  - GitHub: https://github.com/DomSamangy/NBA_Shots_04_25
  - Kaggle: https://www.kaggle.com/datasets/mexwell/nba-shots
- **License:** not stated by the compiler. Underlying data is from NBA.com.
  This repo therefore **does not redistribute any data**: it ships download instructions, a checksum,
  and code only. Use is personal, non-commercial research.
- **Version used:** zip sha256 `edf81d754a6bda8fa3f67c140b4ea4cd45ada8a47569809f28b1f8255eb65d62`.

## Unit and label
- One row = one field-goal attempt (free throws are not in the data).
- Label `made` = 1 if the shot went in. `EVENT_TYPE` restates the label and is forbidden as a feature.

## Known issues found during validation (Oct 2026)
| Issue | Size | Handling |
|---|---|---|
| 2019-20, 2020-21, 2021-22 export `LOC_X`/`LOC_Y` at 1/10 scale with a shifted y origin | ~600k rows (3 seasons) | Rescaled in `nba_shots/clean_data.py`: `x = 10·LOC_X`, `y = 10·(LOC_Y − 5.25)`. Afterwards x/y agree with `SHOT_DISTANCE` to within integer rounding in every season. Guarded by a validation check |
| `GAME_ID` 22000000 is a placeholder shared by five 2020-21 opening games | 873 rows | `game_key` = game_id + date + home team, unique for every game |
| Exact duplicate rows | 192 | Dropped, counted in `results/clean_data_log.csv` |
| Backcourt heaves (end-of-quarter desperation shots, ~2% make rate) | 8,820 | Dropped: this project models normal shot attempts. Counted per season in `results/clean_data_log.csv` |
| Same player, same game, same second (putbacks, tip-ins) | ~5.9k rows | Kept; they are real separate attempts |
| 3PT attempts recorded under 21 ft / 2PT over 24 ft | a few hundred | Kept, counted as warnings |
| Season length varies: 2011-12 lockout (990 games), 2019-20 suspension/bubble, 2020-21 72-game season, 2003-04 has 1,189 of 1,230 games | | Noted; affects per-season counts only |
| `ACTION_TYPE` labels changed over time: "Layup Shot" fell from 15% of shots (2003-12) to 3% (2021-24) while new specific labels ("Cutting Layup Shot", "Driving Floating Jump Shot") appeared around 2012 | all seasons | Main model trains on 2012-13 onward and uses ~9 shot families (`config.SHOT_FAMILIES`) instead of raw labels |
| Shots labeled "Bank" go in far more often than their non-bank versions (57% vs 34% for jump shots), probably because the label is mostly given to shots that banked in | ~2% of shots | Shot families ignore the word "Bank". Dropping it barely changes validation scores |
| `POSITION` is fixed within a player-season but changes across seasons | | Not outcome leakage; treated with caution as a feature |

## Coordinate frame (after cleaning)
`loc_x` runs −25 to 25 across the court (0 = centre), `loc_y` is feet from the baseline (0 to 94).
The hoop centre is at (0, 5.25).

## What the data cannot see
No defender positions, shot clock, dribbles, touch time, or score margin. Shot quality beyond location,
time and shot description is invisible to any model trained here.
