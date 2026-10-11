# Tests use tiny made-up data, so they run anywhere (including GitHub) without the real dataset.
import pandas as pd

from nba_shots.clean_data import clean_season


def fake_shot(season="2018-19", game_id="21800001", date="10-16-2018", home="AAA",
              made=True, x=0.0, y=17.25, distance=12, zone="Mid-Range"):
    """One made-up shot in the same format as the raw CSVs."""
    return {
        "SEASON_1": 2019, "SEASON_2": season, "TEAM_ID": "100", "TEAM_NAME": "Team A",
        "PLAYER_ID": "1", "PLAYER_NAME": "Player One", "POSITION_GROUP": "G", "POSITION": "PG",
        "GAME_DATE": date, "GAME_ID": game_id, "HOME_TEAM": home, "AWAY_TEAM": "BBB",
        "EVENT_TYPE": "Made Shot" if made else "Missed Shot", "SHOT_MADE": made,
        "ACTION_TYPE": "Jump Shot", "SHOT_TYPE": "2PT Field Goal", "BASIC_ZONE": zone,
        "ZONE_NAME": "Center", "ZONE_ABB": "C", "ZONE_RANGE": "8-16 ft.",
        "LOC_X": x, "LOC_Y": y, "SHOT_DISTANCE": distance,
        "QUARTER": 1, "MINS_LEFT": 10, "SECS_LEFT": 30,
    }


def test_scaled_season_coordinates_get_fixed():
    # in a scaled season, a shot 12 ft straight out is saved as y = 17.25 / 10 + 5.25
    raw = pd.DataFrame([fake_shot(season="2020-21", y=17.25 / 10 + 5.25)])
    clean, _ = clean_season(raw)
    assert abs(clean["loc_y"].iloc[0] - 17.25) < 1e-6


def test_normal_season_coordinates_untouched():
    raw = pd.DataFrame([fake_shot(x=-3.0, y=17.25)])
    clean, _ = clean_season(raw)
    assert clean["loc_x"].iloc[0] == -3.0
    assert clean["loc_y"].iloc[0] == 17.25


def test_exact_duplicates_dropped_but_putbacks_kept():
    miss = fake_shot(made=False)
    putback = fake_shot(made=True)  # same player, same second, different result: a real shot
    clean, dropped = clean_season(pd.DataFrame([miss, miss, putback]))
    assert dropped["duplicates_dropped"] == 1
    assert len(clean) == 2


def test_placeholder_game_ids_get_different_keys():
    raw = pd.DataFrame([fake_shot(season="2020-21", game_id="22000000", home="AAA"),
                        fake_shot(season="2020-21", game_id="22000000", home="CCC")])
    clean, _ = clean_season(raw)
    assert clean["game_key"].nunique() == 2


def test_backcourt_heaves_dropped():
    raw = pd.DataFrame([fake_shot(), fake_shot(zone="Backcourt", y=70.0, distance=65, made=False)])
    clean, dropped = clean_season(raw)
    assert dropped["heaves_dropped"] == 1
    assert len(clean) == 1
    assert (clean["basic_zone"] != "Backcourt").all()
