import pandas as pd

from nba_shots.features import add_distance_hinges


def test_hinges_count_feet_past_each_knot():
    df = add_distance_hinges(pd.DataFrame({"shot_distance": [0, 2, 5, 30]}), knots=[1, 3, 23])
    assert list(df["dist_over_1"]) == [0, 1, 4, 29]
    assert list(df["dist_over_3"]) == [0, 0, 2, 27]
    assert list(df["dist_over_23"]) == [0, 0, 0, 7]
