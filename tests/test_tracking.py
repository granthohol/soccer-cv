import os
import pytest
from soccer_cv.pipelines.tracking import write_tracking_video

# Skip cleanly if the sample clip isn't present
clip_path = "content/121364_0.mp4"
skip_msg = f"Sample video not found: {clip_path}"
skip = not os.path.exists(clip_path)

@pytest.mark.skipif(skip, reason=skip_msg)
@pytest.mark.parametrize(
    "input_path, output_path",
    [
        (
            "content/121364_0.mp4",
            "tests/output/tracking_121364_0.mp4",
        ),
    ],
)
def test_track_video_minimal(input_path, output_path):
    # ensure output dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # run the minimal user-facing API (force CPU for portability)
    write_tracking_video(
        input_path,
        output_path
    )

    # metrics path is auto-derived: <output_stem>_metrics.csv
    metrics_path = os.path.splitext(output_path)[0] + "_metrics.csv"

    # simple assertions: files exist and are non-empty
    assert os.path.exists(output_path) and os.path.getsize(output_path) > 0
    assert os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 0
