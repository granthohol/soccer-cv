# src/soccer_cv/pipelines/passing.py
from __future__ import annotations
import os
import csv
from typing import Dict, Tuple

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

try:
    from sports.annotators.soccer import (
        draw_pitch,
        draw_points_on_pitch,
    )
except Exception as e:
    raise ImportError(
        "The 'sports' package is required for this feature. "
        "Install it separately:\n\n"
        "  pip install \"sports @ git+https://github.com/roboflow/sports.git@main\"\n"
    ) from e

from ..config import DEFAULT_CONFIG as CONFIG
from .common import (
    init_runtime,
    detect_ball_and_players,
    classify_players,
    update_homography,
    anchors_bottom_center,
)
from ..utils import nearest_to_ball, PossessionTracker

# ---------------- Tunables ----------------
KEYPOINT_EVERY          = 5      # refresh homography every K frames
OBJ_CONF                = 0.15   # object detection confidence
SMOOTH_H                = 5      # homography smoothing window (frames)
MIN_KP                  = 4      # minimum keypoints to accept a homography
POS_RADIUS_PX           = 100.0  # max pitch distance ball->player to count possession
CONFIRM_FRAMES_DEFAULT  = 3      # consecutive matched frames before a possession change is confirmed
MIN_PRESENCE_FRAMES_DEFAULT = 8  # drop tracks with very few visible frames (noise)
TEAM0_HEX               = "00BFFF"  # blue
TEAM1_HEX               = "FF1493"  # pink
NODE_RADIUS_MIN, NODE_RADIUS_MAX = 14, 30   # node size range, scaled by pass involvement
EDGE_THICK_MIN, EDGE_THICK_MAX   = 2, 14    # edge thickness range, scaled by pass count

# -------------- Canonical pitch bounds (same coords as CONFIG.vertices) --------------
_VERTS = np.asarray(CONFIG.vertices, dtype=np.float32)
X_MIN, X_MAX = float(_VERTS[:, 0].min()), float(_VERTS[:, 0].max())
Y_MIN, Y_MAX = float(_VERTS[:, 1].min()), float(_VERTS[:, 1].max())
_X_SPAN = max(1e-6, X_MAX - X_MIN)
_Y_SPAN = max(1e-6, Y_MAX - Y_MIN)


def _canon_to_img_xy(xy: np.ndarray, template: np.ndarray) -> np.ndarray:
    if xy is None or xy.size == 0:
        return np.empty((0, 2), np.float32)
    h, w = template.shape[:2]
    u = (xy[:, 0] - X_MIN) / _X_SPAN
    v = (xy[:, 1] - Y_MIN) / _Y_SPAN
    x_img = u * (w - 1)
    y_img = v * (h - 1)
    return np.stack([x_img, y_img], axis=1).astype(np.float32)


def _put_label_bgr(img: np.ndarray, text: str, org, color=(255, 255, 255)) -> None:
    """Draw a small bold label (OpenCV BGR)."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 3, cv2.LINE_AA)       # outline
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 1, cv2.LINE_AA)           # fill


def write_pass_network(
    source_video: str,
    output_path: str,
    *,
    min_presence_frames: int = MIN_PRESENCE_FRAMES_DEFAULT,
    confirm_frames: int = CONFIRM_FRAMES_DEFAULT,
    radius_px: float = POS_RADIUS_PX,
) -> None:
    """
    Render a static pass-network image over the canonical 2D pitch.

    Runs a single detect/track/classify pass over the video, accumulating each
    tracked player's average on-pitch position and using a debounced possession
    tracker (see ``soccer_cv.utils.PossessionTracker``) to detect confirmed
    pass/turnover events. Renders one node per player who was visible for at
    least `min_presence_frames` (colored by team, sized by pass involvement,
    positioned at their average location) and one edge per pair of teammates who
    passed to each other (thickness proportional to pass count). Turnovers are
    not drawn as edges.

    Alongside the PNG, writes two structured data files (auto-derived paths, no
    extra parameters needed):
      - ``<output_stem>_nodes.csv``: per-player position/pass/turnover stats
      - ``<output_stem>_edges.csv``: per-pair completed-pass counts

    Parameters
    ----------
    source_video : str
        Path to the input broadcast video.
    output_path : str
        Path to the output PNG.
    min_presence_frames : int, optional
        Minimum number of visible frames for a track to appear as a node. Default: 8.
    confirm_frames : int, optional
        Consecutive matched frames required before a possession change is confirmed.
        Default: 3.
    radius_px : float, optional
        Max pitch distance (canonical units) between ball and player to count as
        possession. Default: 100.0.
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    rt = init_runtime(source_video, want_team_classifier=True)
    if not hasattr(rt, "team_id_map"):
        rt.team_id_map = {}

    frames = sv.get_video_frames_generator(source_video)

    pos_sum: Dict[int, np.ndarray] = {}
    presence: Dict[int, int] = {}
    team_of: Dict[int, int] = {}
    passes_made: Dict[int, int] = {}
    passes_received: Dict[int, int] = {}
    turnovers_committed: Dict[int, int] = {}
    turnovers_conceded: Dict[int, int] = {}
    edge_counts: Dict[Tuple[int, int], int] = {}

    tracker = PossessionTracker(confirm_frames=confirm_frames)

    for i, frame in enumerate(tqdm(frames, total=rt.src_info.total_frames)):
        ball, players, refs = detect_ball_and_players(frame, rt, conf_obj=OBJ_CONF)
        team_ids = classify_players(frame, players, rt.team_classifier, rt.team_id_map, frame_idx=i)

        if (rt.vt is None) or (i % KEYPOINT_EVERY == 0):
            update_homography(frame, rt, keypoint_conf=0.30, min_points=MIN_KP, smooth_len=SMOOTH_H)

        if rt.vt is not None:
            pitch_ball = rt.vt.transform_points(anchors_bottom_center(ball)) if ball.xyxy.size else np.empty((0, 2), np.float32)
            pitch_play = rt.vt.transform_points(anchors_bottom_center(players)) if players.xyxy.size else np.empty((0, 2), np.float32)
        else:
            pitch_ball = np.empty((0, 2), np.float32)
            pitch_play = np.empty((0, 2), np.float32)

        tids = getattr(players, "tracker_id", None)
        if tids is not None and pitch_play.size and len(tids) == len(pitch_play):
            for k in range(len(tids)):
                tid_raw = tids[k]
                if tid_raw is None:
                    continue
                tid = int(tid_raw)
                pos_sum.setdefault(tid, np.zeros(2, dtype=np.float64))
                presence.setdefault(tid, 0)
                pos_sum[tid] += pitch_play[k]
                presence[tid] += 1
                tm = int(team_ids[k]) if k < len(team_ids) else -1
                if tm in (0, 1):
                    team_of[tid] = tm

        poss_res = nearest_to_ball(pitch_ball, pitch_play, team_ids=team_ids, tracker_ids=tids, radius_px=radius_px)
        event = tracker.update(i, poss_res.tid, poss_res.team)
        if event is not None:
            if event.type == "pass":
                a, b = event.from_tid, event.to_tid
                key = (a, b) if a < b else (b, a)
                edge_counts[key] = edge_counts.get(key, 0) + 1
                passes_made[a] = passes_made.get(a, 0) + 1
                passes_received[b] = passes_received.get(b, 0) + 1
            else:
                turnovers_committed[event.from_tid] = turnovers_committed.get(event.from_tid, 0) + 1
                turnovers_conceded[event.to_tid] = turnovers_conceded.get(event.to_tid, 0) + 1

    node_tids = sorted(tid for tid, p in presence.items() if p >= min_presence_frames and tid in team_of)
    avg_pos = {tid: pos_sum[tid] / max(1, presence[tid]) for tid in node_tids}

    template = draw_pitch(CONFIG)
    canvas = template.copy()

    # Edges first, so nodes render on top
    if edge_counts:
        max_count = max(edge_counts.values())
        for (a, b), count in edge_counts.items():
            if a not in avg_pos or b not in avg_pos:
                continue
            pa = _canon_to_img_xy(avg_pos[a].reshape(1, 2), canvas)[0]
            pb = _canon_to_img_xy(avg_pos[b].reshape(1, 2), canvas)[0]
            thickness = int(round(EDGE_THICK_MIN + (EDGE_THICK_MAX - EDGE_THICK_MIN) * (count / max_count)))
            color_hex = TEAM0_HEX if team_of.get(a) == 0 else TEAM1_HEX
            color_bgr = tuple(int(c) for c in sv.Color.from_hex(color_hex).as_bgr())
            cv2.line(canvas, tuple(pa.astype(int)), tuple(pb.astype(int)), color_bgr, thickness, cv2.LINE_AA)

    max_involve = max((passes_made.get(t, 0) + passes_received.get(t, 0) for t in node_tids), default=0) or 1
    for tid in node_tids:
        xy = avg_pos[tid].reshape(1, 2)
        color_hex = TEAM0_HEX if team_of[tid] == 0 else TEAM1_HEX
        involve = passes_made.get(tid, 0) + passes_received.get(tid, 0)
        radius = int(round(NODE_RADIUS_MIN + (NODE_RADIUS_MAX - NODE_RADIUS_MIN) * (involve / max_involve)))
        canvas = draw_points_on_pitch(
            CONFIG, xy,
            face_color=sv.Color.from_hex(color_hex),
            edge_color=sv.Color.BLACK,
            radius=radius, pitch=canvas,
        )
        img_xy = _canon_to_img_xy(xy, canvas)[0]
        _put_label_bgr(canvas, f"#{tid}", org=(int(img_xy[0]) + 8, int(img_xy[1]) - 8))

    cv2.imwrite(output_path, canvas)

    stem = os.path.splitext(output_path)[0]

    with open(f"{stem}_nodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "track_id", "team_id", "x_m", "y_m", "presence_frames",
            "passes_made", "passes_received", "turnovers_committed", "turnovers_conceded",
        ])
        writer.writeheader()
        for tid in node_tids:
            writer.writerow({
                "track_id": tid,
                "team_id": team_of[tid],
                "x_m": round(float(avg_pos[tid][0]) / 100.0, 3),
                "y_m": round(float(avg_pos[tid][1]) / 100.0, 3),
                "presence_frames": presence[tid],
                "passes_made": passes_made.get(tid, 0),
                "passes_received": passes_received.get(tid, 0),
                "turnovers_committed": turnovers_committed.get(tid, 0),
                "turnovers_conceded": turnovers_conceded.get(tid, 0),
            })

    with open(f"{stem}_edges.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tid_a", "tid_b", "team_id", "pass_count"])
        writer.writeheader()
        for (a, b), count in sorted(edge_counts.items(), key=lambda kv: -kv[1]):
            if a not in avg_pos or b not in avg_pos:
                continue
            writer.writerow({
                "tid_a": a, "tid_b": b,
                "team_id": team_of.get(a, -1),
                "pass_count": count,
            })
