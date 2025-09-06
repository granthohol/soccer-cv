# src/soccer_cv/compare.py
from __future__ import annotations
import os
from typing import Optional

import cv2
import numpy as np

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **_: x  # no-op fallback


def _video_info(path: str) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    cap.release()
    return float(fps), w, h, n


def _draw_title(img: np.ndarray, title: Optional[str]) -> None:
    if not title:
        return
    h, w = img.shape[:2]
    pad, bar_h = 8, 28
    # bar
    cv2.rectangle(img, (0, 0), (w, bar_h + 2 * pad), (30, 30, 30), -1)
    # text shadow
    cv2.putText(img, title, (12, pad + bar_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    # text
    cv2.putText(img, title, (12, pad + bar_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def write_side_by_side_video(
    left_video: str,
    right_video: str,
    out_video: str,
    *,
    left_title: str = "Input",
    right_title: str = "Output",
    output_fps: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> None:
    """
    Create a side-by-side comparison video from two videos (e.g., input vs. Voronoi).

    - Resizes each side to the same HEIGHT (max of the two) while preserving aspect ratio.
    - Uses min available frames so it never overruns a shorter stream.
    """
    fps_l, wl, hl, nl = _video_info(left_video)
    fps_r, wr, hr, nr = _video_info(right_video)

    fps = float(output_fps or fps_l or fps_r or 30.0)
    total = min([x for x in [nl, nr, max_frames] if (x is not None and x > 0)] or [max(nl, nr)])

    cap_l = cv2.VideoCapture(left_video)
    cap_r = cv2.VideoCapture(right_video)
    assert cap_l.isOpened() and cap_r.isOpened()

    # common output height = max of both; compute widths preserving aspect
    H = max(hl, hr)
    new_wl = int(round(wl * (H / hl)))
    new_wr = int(round(wr * (H / hr)))
    W = new_wl + new_wr

    os.makedirs(os.path.dirname(out_video) or ".", exist_ok=True)
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    for i in tqdm(range(total), total=total, desc="Side-by-side"):
        ok_l, fl = cap_l.read()
        ok_r, fr = cap_r.read()
        if not ok_l or not ok_r:
            break

        fl = cv2.resize(fl, (new_wl, H), interpolation=cv2.INTER_AREA)
        fr = cv2.resize(fr, (new_wr, H), interpolation=cv2.INTER_AREA)

        _draw_title(fl, left_title)
        _draw_title(fr, right_title)

        combo = np.concatenate([fl, fr], axis=1)
        writer.write(combo)

    cap_l.release()
    cap_r.release()
    writer.release()


def write_video_with_image(
    video_path: str,
    image_path: str,
    out_video: str,
    *,
    side: str = "right",               # "right" or "left"
    video_title: str = "Input",
    image_title: str = "Heatmap",
    panel_width: Optional[int] = None, # by default scales the image to video height
    output_fps: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> None:
    """
    Display a video next to a static image (e.g., team/player heatmap .png).
    The image is repeated for every frame so the duration matches the video.
    """
    fps, w, h, n = _video_info(video_path)
    fps = float(output_fps or fps or 30.0)
    total = min([x for x in [n, max_frames] if (x is not None and x > 0)] or [n])

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # scale image to same height; optional override width
    scale = h / img.shape[0]
    new_w = panel_width or int(round(img.shape[1] * scale))
    panel = cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA)

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()

    if side.lower() == "right":
        W = w + new_w
    else:
        W = new_w + w

    os.makedirs(os.path.dirname(out_video) or ".", exist_ok=True)
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, h))

    # Pre-title the static panel once
    panel_anno = panel.copy()
    _draw_title(panel_anno, image_title)

    for i in tqdm(range(total), total=total, desc="Video+Image"):
        ok, frame = cap.read()
        if not ok:
            break

        _draw_title(frame, video_title)

        if side.lower() == "right":
            combo = np.concatenate([frame, panel_anno], axis=1)
        else:
            combo = np.concatenate([panel_anno, frame], axis=1)
        writer.write(combo)

    cap.release()
    writer.release()
