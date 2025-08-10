# src/soccer_cv/__init__.py
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("soccer-cv")
except PackageNotFoundError:  # when running from source without install
    __version__ = "0.0.0"

# Public API
from .pipelines.ball_path import write_ball_path_2d_video
from .pipelines.voronoi import write_voronoi_2d_video

__all__ = [
    "write_ball_path_2d_video",
    "write_voronoi_2d_video",
    "__version__",
]