import pytest
from soccer_cv.pipelines.possession import write_possession_2d_video
import os

@pytest.mark.parametrize("input_path, output_path", [
    ("content/08fd33_0.mp4", "tests/output/possession_08fd33_0.mp4"),
])
def test_write_possession_2d_video(input_path, output_path):
    # ensure output dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # call function under test
    write_possession_2d_video(input_path, output_path)

    # simple assertion: output file should exist
    assert os.path.exists(output_path)