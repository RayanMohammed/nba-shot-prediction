# pytest runs this file before the tests. It hides two kinds of harmless warnings that come from
# inside other libraries (not our code), so the test output stays readable:
#  - Apple-chip false alarms "... encountered in matmul" (same as nba_shots/__init__.py; pytest
#    resets warning filters for each test, so the one in __init__.py doesn't reach here)
#  - matplotlib calling old pyparsing function names


def pytest_configure(config):
    config.addinivalue_line("filterwarnings", "ignore:.*encountered in matmul:RuntimeWarning")
    config.addinivalue_line("filterwarnings", "ignore::DeprecationWarning:matplotlib.*")
