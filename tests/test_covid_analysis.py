import numpy as np
import pandas as pd

from nba_shots.covid_analysis import gap_and_home_edge, per_game_table


def test_gap_and_home_edge_by_hand():
    # one game: 2 home shots (both made), 2 away shots (both missed); model predicts 50% for all
    df = pd.DataFrame({"game_key": "g1", "is_home": [1, 1, 0, 0], "made": [1, 1, 0, 0]})
    games = per_game_table(df, np.full(4, 0.5))
    gap, home_edge = gap_and_home_edge(games)
    assert np.isclose(gap, 0.0)          # 50% made overall, 50% predicted
    assert np.isclose(home_edge, 1.0)    # home beat the model by +0.5 per shot, away by -0.5
