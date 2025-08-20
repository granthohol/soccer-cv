import pytest
from soccer_cv.pipelines.ball_path import write_ball_path_2d_video
import os

@pytest.mark.parametrize("input_path, output_path", [
    ("content/121364_0.mp4", "tests/output/ball_path_121364_0.mp4"),
])
def test_write_voronoi_2d_video(input_path, output_path):
    # ensure output dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # call function under test
    write_ball_path_2d_video(input_path, output_path)

    # simple assertion: output file should exist
    assert os.path.exists(output_path)