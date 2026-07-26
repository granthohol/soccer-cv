import os
import sys
import subprocess

import pytest

from soccer_cv import __version__
from soccer_cv.cli import main

# Skip cleanly if the sample clip isn't present
clip_path = "content/121364_0.mp4"
skip_msg = f"Sample video not found: {clip_path}"
skip = not os.path.exists(clip_path)

SUBCOMMANDS = [
    "tracking", "team-shape", "voronoi", "ball-path", "heatmaps",
    "heatmap-grids", "possession", "pass-network", "summarize",
    "compare-side-by-side", "compare-with-image",
]


def test_top_level_help():
    result = subprocess.run(
        [sys.executable, "-m", "soccer_cv.cli", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


@pytest.mark.parametrize("subcommand", SUBCOMMANDS)
def test_subcommand_help(subcommand):
    result = subprocess.run(
        [sys.executable, "-m", "soccer_cv.cli", subcommand, "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def test_version_flag():
    result = subprocess.run(
        [sys.executable, "-m", "soccer_cv.cli", "--version"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert __version__ in result.stdout


def test_missing_command_exits_nonzero():
    result = subprocess.run(
        [sys.executable, "-m", "soccer_cv.cli"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0


@pytest.mark.skipif(skip, reason=skip_msg)
def test_team_shape_end_to_end():
    output_path = "tests/output/cli_team_shape_121364_0.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rc = main(["team-shape", "content/121364_0.mp4", output_path])

    assert rc == 0
    assert os.path.exists(output_path) and os.path.getsize(output_path) > 0
