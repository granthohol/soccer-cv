import pytest
from soccer_cv.pipelines.player_heatmaps import write_team_heatmaps_video, write_team_player_heatmap_grids
import os

# Skip cleanly if the sample clip isn't present
clip_path = "content/121364_0.mp4"
skip_msg = f"Sample video not found: {clip_path}"
skip = not os.path.exists(clip_path)

@pytest.mark.skipif(skip, reason=skip_msg)
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

    # heatmaps path is auto-derived: <output_stem>_heatmaps.npz
    heatmaps_path = os.path.splitext(output_path)[0] + "_heatmaps.npz"
    assert os.path.exists(heatmaps_path) and os.path.getsize(heatmaps_path) > 0

@pytest.mark.skipif(skip, reason=skip_msg)
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

    # structured data exports, written alongside the PNGs in output_dir
    npz_path = os.path.join(output_dir, "player_heatmaps.npz")
    manifest_path = os.path.join(output_dir, "player_heatmaps_manifest.csv")
    assert os.path.isfile(npz_path) and os.path.getsize(npz_path) > 0
    assert os.path.isfile(manifest_path) and os.path.getsize(manifest_path) > 0
