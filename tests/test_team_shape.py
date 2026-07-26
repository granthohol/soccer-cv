import pytest
from soccer_cv.pipelines.team_shape import write_team_shape_video
import os

# Skip cleanly if the sample clip isn't present
clip_path = "content/121364_0.mp4"
skip_msg = f"Sample video not found: {clip_path}"
skip = not os.path.exists(clip_path)

@pytest.mark.skipif(skip, reason=skip_msg)
@pytest.mark.parametrize("input_path, output_path", [
    ("content/121364_0.mp4", "tests/output/team_shape_121364_0.mp4"),
])
def test_write_team_shape_2d_video(input_path, output_path):
    # ensure output dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # call function under test
    write_team_shape_video(input_path, output_path)

    # simple assertion: output file should exist
    assert os.path.exists(output_path)

    # metrics path is auto-derived: <output_stem>_metrics.csv
    metrics_path = os.path.splitext(output_path)[0] + "_metrics.csv"
    assert os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 0
