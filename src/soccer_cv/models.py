# src/soccer_cv/models.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

CACHE_DIR = Path.home() / ".cache" / "soccer-cv"

@dataclass(frozen=True)
class ModelSpec:
    repo_id: str          # e.g. "your-org/soccer-cv"
    filename: str         # e.g. "players_yolo11n.pt"
    sha256: str           # fill with your file's sha256

OBJECT_SPEC = ModelSpec(
    repo_id="granthohol/soccer-cv-weights", 
    filename="models/object_detection/weights/best_object_detection.pt", 
    sha256="30069e51c37f7e89cc2f3d692aee967ccbf3dd3fb815b3cd294ca04c7eee6931"
    )
PITCH_SPEC  = ModelSpec(
    repo_id="granthohol/soccer-cv-weights", 
    filename="models/pitch_detection/weights/best_pitch_detection.pt", 
    sha256="d9594ce626ebba50eb8fdf7d8118c17e9ddaf1f7a186eb088106f6c6b79b8b0b"
    )

def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _ensure_weights(spec: ModelSpec) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = hf_hub_download(repo_id=spec.repo_id, filename=spec.filename, local_dir=CACHE_DIR)
    got = _sha256(local_path)
    if spec.sha256 != "CHANGE_ME_SHA256" and got != spec.sha256:
        raise RuntimeError(f"Checksum mismatch for {spec.filename}: {got} != {spec.sha256}")
    return local_path

def load_default_object_model() -> YOLO:
    return YOLO(_ensure_weights(OBJECT_SPEC))

def load_default_pitch_model() -> YOLO:
    return YOLO(_ensure_weights(PITCH_SPEC))
