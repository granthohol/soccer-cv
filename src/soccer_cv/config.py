# src/soccer_cv/config.py
try:
    from sports.configs.soccer import (
        SoccerPitchConfiguration
    )
except Exception as e:
    raise ImportError(
        "The 'sports' package is required for this feature. "
        "Install it separately:\n\n"
        "  pip install \"sports @ git+https://github.com/roboflow/sports.git@main\"\n"
    ) from e

# expose a single default config the library uses internally
DEFAULT_CONFIG = SoccerPitchConfiguration()
