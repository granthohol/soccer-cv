import pytest
from soccer_cv.pipelines.passing import write_pass_network
import os

# Skip cleanly if the sample clip isn't present
clip_path = "content/121364_0.mp4"
skip_msg = f"Sample video not found: {clip_path}"
skip = not os.path.exists(clip_path)

@pytest.mark.skipif(skip, reason=skip_msg)
@pytest.mark.parametrize("input_path, output_path", [
    ("content/121364_0.mp4", "tests/output/pass_network_121364_0.png"),
])
def test_write_pass_network(input_path, output_path):
    # ensure output dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # call function under test
    write_pass_network(input_path, output_path)

    # simple assertion: output file should exist
    assert os.path.exists(output_path) and os.path.getsize(output_path) > 0

    # structured data exports, auto-derived from output_path's stem
    stem = os.path.splitext(output_path)[0]
    nodes_path = f"{stem}_nodes.csv"
    edges_path = f"{stem}_edges.csv"
    assert os.path.exists(nodes_path) and os.path.getsize(nodes_path) > 0
    assert os.path.exists(edges_path) and os.path.getsize(edges_path) > 0
