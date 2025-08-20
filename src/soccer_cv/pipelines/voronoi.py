# src/soccer_cv/pipelines/voronoi2d.py
from __future__ import annotations
import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

from sports.annotators.soccer import (
    draw_pitch,
    draw_points_on_pitch,
    draw_pitch_voronoi_diagram,
)

from ..config import DEFAULT_CONFIG as CONFIG
from .common import (
    init_runtime, detect_ball_and_players, classify_players,
    update_homography, anchors_bottom_center,
)

# Tunables
VORONOI_EVERY   = 5     # compute Voronoi polygons every K frames
KEYPOINT_EVERY  = 5     # refresh homography every K' frames
MIN_KP          = 4
OBJ_CONF        = 0.15
SMOOTH_H        = 5
TEAM1_HEX       = '00BFFF'
TEAM2_HEX       = 'FF1493'


def _make_voronoi_layer(
    template: np.ndarray,
    team1_xy: np.ndarray,
    team2_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (layer, mask) where:
      - layer is an HxWx3 uint8 image with ONLY the Voronoi colors on black
      - mask is a boolean HxW array where Voronoi paint exists
    We draw on a black canvas so we can later composite just the polygons.
    """
    h, w = template.shape[:2]
    layer = np.zeros_like(template, dtype=np.uint8)  # black background

    layer = draw_pitch_voronoi_diagram(
        config=CONFIG,
        team_1_xy=team1_xy,
        team_2_xy=team2_xy,
        team_1_color=sv.Color.from_hex(TEAM1_HEX),
        team_2_color=sv.Color.from_hex(TEAM2_HEX),
        pitch=layer
    )
    mask = np.any(layer != 0, axis=2)  # painted pixels
    return layer, mask


def _blend_layers(prev_layer: np.ndarray, curr_layer: np.ndarray, alpha: float) -> np.ndarray:
    """Linear blend between two uint8 layers (HxWx3)."""
    return cv2.addWeighted(prev_layer, 1.0 - alpha, curr_layer, alpha, 0.0)


def write_voronoi_2d_video(source_video: str, target_video: str) -> None:
    """
    Fast Voronoi: blend only the Voronoi polygons between keyframes.
    Player/ball points are drawn freshly each frame (no blending).
    """
    max_frames = int(os.getenv("SOCCER_CV_MAX_FRAMES", "0")) or None

    rt = init_runtime(source_video, want_team_classifier=True)
    frames = sv.get_video_frames_generator(source_video)

    prev_layer = None         # np.ndarray (HxWx3)
    prev_mask  = None         # np.ndarray (HxW) bool
    curr_layer = None
    curr_mask  = None
    blend_t = 0
    blend_steps = max(1, VORONOI_EVERY)

    with sv.VideoSink(target_video, video_info=rt.pitch_info) as sink:
        for i, frame in enumerate(tqdm(frames, total=rt.src_info.total_frames)):
            if max_frames is not None and i >= max_frames:
                break

            # Always detect & classify per frame so points stay fresh
            ball, players, refs = detect_ball_and_players(frame, rt, conf_obj=OBJ_CONF)
            team_ids = classify_players(frame, players, rt.team_classifier, rt.team_id_map, frame_idx=i)

            # Update homography periodically (independent of Voronoi cadence)
            if (rt.vt is None) or (i % KEYPOINT_EVERY == 0):
                update_homography(frame, rt, keypoint_conf=0.30, min_points=MIN_KP, smooth_len=SMOOTH_H)

            # Project anchors if homography available
            if rt.vt is not None:
                pitch_ball = rt.vt.transform_points(anchors_bottom_center(ball))
                pitch_play = rt.vt.transform_points(anchors_bottom_center(players))
                pitch_refs = rt.vt.transform_points(anchors_bottom_center(refs))
            else:
                pitch_ball = np.empty((0, 2), np.float32)
                pitch_play = np.empty((0, 2), np.float32)
                pitch_refs = np.empty((0, 2), np.float32)

            is_keyframe = (i % VORONOI_EVERY == 0)
            if is_keyframe:
                # Build a NEW Voronoi layer on keyframes only
                if rt.vt is None or pitch_play.size == 0:
                    new_layer = np.zeros_like(rt.template)
                    new_mask  = np.zeros(rt.template.shape[:2], dtype=bool)
                else:
                    new_layer, new_mask = _make_voronoi_layer(
                        rt.template,
                        team1_xy=pitch_play[team_ids == 0],
                        team2_xy=pitch_play[team_ids == 1],
                    )

                if curr_layer is None:
                    # First keyframe: nothing to blend from
                    curr_layer, curr_mask = new_layer, new_mask
                    prev_layer, prev_mask = new_layer.copy(), new_mask.copy()
                    blend_t = blend_steps  # skip blending on first output
                    voronoi_composited = curr_layer
                    active_mask = curr_mask
                else:
                    # Start a new transition: prev <- curr, curr <- new
                    prev_layer, prev_mask = curr_layer, curr_mask
                    curr_layer, curr_mask = new_layer, new_mask
                    blend_t = 0
                    voronoi_composited = curr_layer
                    active_mask = curr_mask
            else:
                # Non-keyframe: cross-fade Voronoi ONLY
                if prev_layer is None or curr_layer is None:
                    voronoi_composited = np.zeros_like(rt.template)
                    active_mask = np.zeros(rt.template.shape[:2], dtype=bool)
                else:
                    blend_t = min(blend_t + 1, blend_steps)
                    alpha = blend_t / float(blend_steps)
                    blended = _blend_layers(prev_layer, curr_layer, alpha)
                    # where to draw: union of both masks (so fading out old polygons works)
                    active_mask = (prev_mask | curr_mask)
                    voronoi_composited = blended

            # Compose: base pitch + blended Voronoi polygons (masked), then draw points fresh
            canvas = rt.template.copy()
            if active_mask.any():
                # alpha-blend voronoi onto the pitch only at polygon pixels
                a = float(0.5)
                # do math in float, write back to uint8
                bg = canvas[active_mask].astype(np.float32)
                fg = voronoi_composited[active_mask].astype(np.float32)
                blended = (1.0 - a) * bg + a * fg
                canvas[active_mask] = blended.astype(np.uint8)

            # Draw players / ball / refs for THIS frame
            if pitch_play.size:
                canvas = draw_points_on_pitch(
                    CONFIG, pitch_play[team_ids == 0],
                    face_color=sv.Color.from_hex(TEAM1_HEX),
                    edge_color=sv.Color.BLACK,
                    radius=14, pitch=canvas
                )
                canvas = draw_points_on_pitch(
                    CONFIG, pitch_play[team_ids == 1],
                    face_color=sv.Color.from_hex(TEAM2_HEX),
                    edge_color=sv.Color.BLACK,
                    radius=14, pitch=canvas
                )
            if pitch_refs.size:
                canvas = draw_points_on_pitch(
                    CONFIG, pitch_refs,
                    face_color=sv.Color.BLACK, edge_color=sv.Color.WHITE,
                    radius=16, pitch=canvas
                )
            if pitch_ball.size:
                canvas = draw_points_on_pitch(
                    CONFIG, pitch_ball,
                    face_color=sv.Color.WHITE, edge_color=sv.Color.BLACK,
                    radius=10, pitch=canvas
                )

            sink.write_frame(canvas)
