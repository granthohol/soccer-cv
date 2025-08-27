import pytest
from soccer_cv.pipelines.player_heatmaps import write_team_heatmaps_video, write_team_player_heatmap_grids
import os

@pytest.mark.parametrize("input_path, output_path", [
    ("content/121364_0.mp4", "tests/output/heatmaps_121364_0.mp4"),
])
def test_write_heatmaps_2d_video(input_path, output_path):
    # ensure output dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # call function under test
    write_team_heatmaps_video(input_path, output_path)

    # simple assertion: output file should exist
    assert os.path.exists(output_path)
    
@pytest.mark.parametrize("input_path, output_dir", [
    ("content/121364_0.mp4", "tests/output/heatmaps_grid"),
])
def test_write_team_player_heatmap_grids(input_path, output_dir):
    # ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    # call function under test (returns two PNG paths)
    team0_path, team1_path = write_team_player_heatmap_grids(
        input_path,
        output_dir,
    )

    # assertions: files exist and are non-empty
    assert os.path.isfile(team0_path)
    assert os.path.isfile(team1_path)
    assert os.path.getsize(team0_path) > 0
    assert os.path.getsize(team1_path) > 0