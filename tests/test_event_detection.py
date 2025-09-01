import pytest
from soccer_cv.pipelines.events import detect_events_to_file
import os

@pytest.mark.parametrize("input_path, output_path", [
    ("content/121364_0.mp4", "tests/output/events_121364_0.csv"),
])
def test_detect_events_to_file_csv(input_path, output_path):
    # ensure output dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # call function under test
    detect_events_to_file(input_path, output_path)

    # simple assertion: output file should exist
    assert os.path.exists(output_path)
    
@pytest.mark.parametrize("input_path, output_path", [
    ("content/121364_0.mp4", "tests/output/events_121364_0.jsonl"),
])
def test_detect_events_to_file_json(input_path, output_path):
    # ensure output dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # call function under test
    detect_events_to_file(input_path, output_path, fmt="jsonl")

    # simple assertion: output file should exist
    assert os.path.exists(output_path)