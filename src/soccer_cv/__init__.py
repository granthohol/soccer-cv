# src/soccer_cv/__init__.py
import os
from importlib.metadata import PackageNotFoundError, version

# Quiet down third-party logging (transformers/HF) before any heavy imports pull them in
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    __version__ = version("soccer-cv")
except PackageNotFoundError:  # when running from source without install
    __version__ = "1.0.0"

# Public API
from .pipelines.ball_path import write_ball_path_2d_video
from .pipelines.voronoi import write_voronoi_2d_video
from .pipelines.player_heatmaps import write_team_heatmaps_video, write_team_player_heatmap_grids
from .pipelines.possession import write_possession_2d_video
from .pipelines.passing import write_pass_network
from .pipelines.team_shape import write_team_shape_video
from .pipelines.tracking import write_tracking_video, summarize_player_stats
from .compare import write_side_by_side_video, write_video_with_image

__all__ = [
    "__version__",
    "write_ball_path_2d_video",
    "write_voronoi_2d_video",
    "write_team_heatmaps_video",
    "write_team_player_heatmap_grids",
    "write_possession_2d_video",
    "write_pass_network",
    "write_team_shape_video",
    "write_tracking_video",
    "summarize_player_stats",
    "write_side_by_side_video",
    "write_video_with_image",
]