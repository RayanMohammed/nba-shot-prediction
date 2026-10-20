# Columns that can never be used as model inputs, because they give away the answer
# or are only known after the shot / game. Every feature list gets checked against this.

FORBIDDEN = {
    "made": "this is the label",
    "event_type": "says 'Made Shot' or 'Missed Shot', so it's the label again",
    "shot_made": "the label (original name)",
    "pts": "points scored on the shot",
    "fgm": "the label (shot-logs name)",
    "shot_result": "the label (shot-logs name)",
    "w": "game result, only known after the game",
    "final_margin": "only known after the game",
}

# Allowed, but we must also report results WITHOUT these, since they're written down after the shot.
# action_type may also be partly tied to the result: "Bank" seems to appear mostly on shots that banked IN.
# The main model only uses shot families (config.SHOT_FAMILIES), which ignore the word "Bank".
FLAGGED = ["action_type"]

# Allowed ONLY because they're built from earlier games (features.add_shooter_history), never from the
# shot's own game or later ones. tests/test_features.py checks that.
AS_OF_ONLY = ["shooter_skill", "shooter_prior_shots", "shooter_prior_resid",
              "opponent_defense", "opponent_prior_shots", "opponent_prior_resid"]


def check_features(feature_cols):
    """Raise an error if any feature is on the forbidden list."""
    bad = [col for col in feature_cols if col.lower() in FORBIDDEN]
    if len(bad) > 0:
        reasons = [col + " (" + FORBIDDEN[col.lower()] + ")" for col in bad]
        raise ValueError("Forbidden feature columns: " + ", ".join(reasons))
