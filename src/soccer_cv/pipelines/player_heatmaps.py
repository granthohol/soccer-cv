# src/soccer_cv/pipelines/heatmaps.py
from __future__ import annotations
import os
from typing import Deque, Optional
from collections import deque

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch

from ..config import DEFAULT_CONFIG as CONFIG
from .common import (
    init_runtime,
    detect_ball_and_players,
    classify_players,
    update_homography,
    anchors_bottom_center,
)

GRID_W, GRID_H     = 200, 130   # heat grid resolution (columns x rows)
BLUR_SIGMA         = 5.0        # Gaussian sigma for heat smoothing (in grid pixels)
HEAT_ALPHA         = 0.65       # max alpha to blend heat onto the pitch
KEYPOINT_EVERY     = 5          # refresh homography every K frames
SMOOTH_H           = 5          # homography smoothing window
MIN_KP             = 4          # min keypoints for a valid homography
OBJ_CONF           = 0.15       # object detection confidence threshold
TEAM0_HEX          = "00BFFF"   # team 0 “blue”
TEAM1_HEX          = "FF1493"   # team 1 “pink”


# derive canonical pitch bounds from CONFIG.vertices
_VERTS = np.asarray(CONFIG.vertices, dtype=np.float32)  # shape (N, 2)
X_MIN, X_MAX = float(_VERTS[:, 0].min()), float(_VERTS[:, 0].max())
Y_MIN, Y_MAX = float(_VERTS[:, 1].min()), float(_VERTS[:, 1].max())
# Guard division-by-zero if someone passes a degenerate config
_X_SPAN = max(1e-6, X_MAX - X_MIN)
_Y_SPAN = max(1e-6, Y_MAX - Y_MIN)

def _accumulate_heat(grid: np.ndarray, xy_canon: np.ndarray) -> None:
    """
    Add counts into a (GRID_H x GRID_W) heat grid from canonical pitch coords.
    xy_canon is expected to be in the same coordinate system as CONFIG.vertices.
    """
    if xy_canon is None or xy_canon.size == 0:
        return

    # Normalize canonical coords → [0, 1] within the pitch bounds
    u = (xy_canon[:, 0] - X_MIN) / _X_SPAN
    v = (xy_canon[:, 1] - Y_MIN) / _Y_SPAN

    # Convert to grid indices (col,row) and clip
    gx = np.clip((u * GRID_W).astype(np.int32), 0, GRID_W - 1)
    gy = np.clip((v * GRID_H).astype(np.int32), 0, GRID_H - 1)

    np.add.at(grid, (gy, gx), 1.0)
    
def _render_heat_overlay_colormap(
    base_pitch: np.ndarray,
    grid: np.ndarray,
    *,
    frames_seen: int,
    blur_sigma: float = BLUR_SIGMA,
    exposure_pct: int = 98,
    cmap_name: str = "JET",
    alpha_max: float = 0.7,
    gain: float = 1.15,
    gamma: float = 1.2,
) -> np.ndarray:
    """
    Convert a cumulative grid to a colored heatmap and alpha-blend onto the pitch.

    Pipeline:
      1) Smooth grid in grid-space (Gaussian).
      2) Convert to occupancy fraction: occ = grid / frames_seen.
      3) Dynamic exposure: occ /= percentile(occ, exposure_pct) to keep early frames visible.
      4) Perceptual shaping: shaped = clip(gain * occ**gamma, 0..1).
      5) Upsample to pitch size and apply OpenCV colormap (e.g., JET).
      6) Alpha = shaped * alpha_max; out = pitch*(1-alpha) + heat*alpha.
    """
    base = np.ascontiguousarray(base_pitch[..., :3]).astype(np.uint8)
    h, w = base.shape[:2]

    if grid is None or grid.size == 0 or frames_seen <= 0:
        return base.copy()

    # 1) smooth in grid space
    g = cv2.GaussianBlur(grid.astype(np.float32), (0, 0), blur_sigma)

    # 2) occupancy fraction (time-share so far)
    occ = g / float(frames_seen)
    occ = np.nan_to_num(occ, nan=0.0, posinf=0.0, neginf=0.0)

    # 3) dynamic exposure
    if exposure_pct is not None and 0 < exposure_pct < 100:
        ref = float(np.percentile(occ, exposure_pct))
        if ref > 1e-6:
            occ = np.clip(occ / ref, 0.0, 1.0)

    # 4) perceptual shaping
    with np.errstate(invalid="ignore"):
        shaped = gain * np.power(np.clip(occ, 0.0, 1.0), gamma)
    shaped = np.clip(shaped, 0.0, 1.0)

    # 5) to pitch size, apply colormap
    norm_map = cv2.resize(shaped, (w, h), interpolation=cv2.INTER_LINEAR)
    norm_u8  = (norm_map * 255.0).astype(np.uint8)

    # pick a colormap
    # JET → blue→green→yellow→red (red hottest)
    cmap_const = getattr(cv2, f"COLORMAP_{cmap_name.upper()}", cv2.COLORMAP_JET)
    heat_bgr  = cv2.applyColorMap(norm_u8, cmap_const)  # HxWx3 (BGR)

    # 6) alpha blend
    alpha = (norm_map * float(alpha_max))[..., None].astype(np.float32)  # HxWx1
    out   = base.astype(np.float32) * (1.0 - alpha) + heat_bgr.astype(np.float32) * alpha
    return out.astype(np.uint8)


def _concat_horiz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Concatenate two equal-height images horizontally."""
    assert left.shape[:2] == right.shape[:2], "Panels must have same HxW"
    return np.concatenate([left, right], axis=1)

def write_team_heatmaps_video(source_video: str, target_video: str) -> None:
    """
    Render cumulative player heatmaps per team on canonical 2D pitches (split screen) and write a video.

    Parameters
    ----------
    source_video : str
        Path to the input broadcast video.
    target_video : str
        Path to the output video file (MP4 recommended).

    Notes
    -----
    - Heatmaps are cumulative for the entire clip (not rolling).
    """    
    # Initialize runtime (models, device, tracker, team classifier, pitch sizes)
    rt = init_runtime(source_video, want_team_classifier=True)
    if not hasattr(rt, "team_id_map"):
        rt.team_id_map = {}  # ensure exists for classifier caching

    frames = sv.get_video_frames_generator(source_video)

    # Pitch canvas and sizes
    pitch_template = rt.template
    ph, pw = pitch_template.shape[:2]

    # Two cumulative heat grids: team 0, team 1
    heat0 = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    heat1 = np.zeros((GRID_H, GRID_W), dtype=np.float32)

    # Output video is split-screen: width doubles
    split_info = sv.VideoInfo(
        fps=rt.pitch_info.fps,
        width=pw * 2,
        height=ph,
        total_frames=rt.pitch_info.total_frames
    )

    frames_seen = 0

    with sv.VideoSink(target_video, video_info=split_info) as sink:
        for i, frame in enumerate(tqdm(frames, total=rt.src_info.total_frames)):

            # 1) Detect & classify players
            ball, players, refs = detect_ball_and_players(frame, rt, conf_obj=OBJ_CONF)
            team_ids = classify_players(frame, players, rt.team_classifier, rt.team_id_map, frame_idx=i)

            # 2) Update homography periodically
            if (rt.vt is None) or (i % KEYPOINT_EVERY == 0):
                update_homography(frame, rt, keypoint_conf=0.30, min_points=MIN_KP, smooth_len=SMOOTH_H)


            # 3) Accumulate team heat in canonical coords
            pitch_players = np.empty((0, 2), np.float32)
            pitch_ball    = np.empty((0, 2), np.float32)

            if rt.vt is not None:
                if players.xyxy.size:
                    pitch_players = rt.vt.transform_points(anchors_bottom_center(players))
                    if pitch_players.size:
                        mask0 = (team_ids == 0)
                        mask1 = (team_ids == 1)
                        _accumulate_heat(heat0, pitch_players[mask0])
                        _accumulate_heat(heat1, pitch_players[mask1])

                if ball.xyxy.size:
                    b = anchors_bottom_center(ball)
                    if b.size:
                        pitch_ball = rt.vt.transform_points(b)

            # 4) Render each team panel with “time-share so far” normalization
            frames_seen += 1  # we are producing one output frame now
            left_panel  = _render_heat_overlay_colormap(pitch_template, heat0, frames_seen=frames_seen)
            right_panel = _render_heat_overlay_colormap(pitch_template, heat1, frames_seen=frames_seen)

            # Optional live dots for context
            if pitch_players.size:
                if np.any(team_ids == 0):
                    left_panel = draw_points_on_pitch(
                        CONFIG, pitch_players[team_ids == 0],
                        face_color=sv.Color.from_hex(TEAM0_HEX),
                        edge_color=sv.Color.BLACK,
                        radius=12, pitch=left_panel
                    )
                if np.any(team_ids == 1):
                    right_panel = draw_points_on_pitch(
                        CONFIG, pitch_players[team_ids == 1],
                        face_color=sv.Color.from_hex(TEAM1_HEX),
                        edge_color=sv.Color.BLACK,
                        radius=12, pitch=right_panel
                    )

            # Draw the ball (white) on both panels if we have it
            if pitch_ball.size:
                for panel in (left_panel, right_panel):
                    draw_points_on_pitch(
                        CONFIG, pitch_ball,
                        face_color=sv.Color.WHITE, edge_color=sv.Color.BLACK,
                        radius=10, pitch=panel
                    )

            # 5) Write split frame
            split = _concat_horiz(left_panel, right_panel)
            sink.write_frame(split)