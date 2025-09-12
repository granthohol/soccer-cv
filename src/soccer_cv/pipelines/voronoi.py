# src/soccer_cv/pipelines/voronoi2d.py
from __future__ import annotations
import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

try:
    from sports.annotators.soccer import (
        draw_points_on_pitch,
        draw_pitch_voronoi_diagram,
    )
except Exception as e:
    raise ImportError(
        "The 'sports' package is required for this feature. "
        "Install it separately:\n\n"
        "  pip install \"sports @ git+https://github.com/roboflow/sports.git@main\"\n"
    ) from e

from ..config import DEFAULT_CONFIG as CONFIG
from .common import (
    init_runtime, detect_ball_and_players, classify_players,
    update_homography, anchors_bottom_center,
)

# ---------------- Tunables ----------------
VORONOI_EVERY    = 5      # recompute Voronoi polygons every K frames; blend between keyframes
KEYPOINT_EVERY   = 5      # refresh homography every K' frames
MIN_KP           = 4      # minimum keypoints to accept a homography
OBJ_CONF         = 0.15   # object detector confidence
SMOOTH_H         = 5      # homography smoothing window (frames)
TEAM0_HEX        = "00BFFF"  # team_id == 0 (blue/cyan)
TEAM1_HEX        = "FF1493"  # team_id == 1 (pink)
POLY_BLEND_ALPHA = 0.50      # polygon layer → pitch blend
CONTROL_STEP_PX  = 8         # grid step (in output pixels) for control % sampling

# -------------- Canonical pitch bounds (same coords as CONFIG.vertices) --------------
_VERTS = np.asarray(CONFIG.vertices, dtype=np.float32)
X_MIN, X_MAX = float(_VERTS[:, 0].min()), float(_VERTS[:, 0].max())
Y_MIN, Y_MAX = float(_VERTS[:, 1].min()), float(_VERTS[:, 1].max())


# ---------------- Helpers ----------------
def _make_voronoi_layer(
    template: np.ndarray,
    team0_xy: np.ndarray,
    team1_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (layer, mask) where:
      - layer is an HxWx3 uint8 image with ONLY the Voronoi colors on black
      - mask is a boolean HxW array where Voronoi paint exists
    """
    layer = np.zeros_like(template, dtype=np.uint8)
    layer = draw_pitch_voronoi_diagram(
        config=CONFIG,
        team_1_xy=team0_xy,  # our team_id==0
        team_2_xy=team1_xy,  # our team_id==1
        team_1_color=sv.Color.from_hex(TEAM0_HEX),
        team_2_color=sv.Color.from_hex(TEAM1_HEX),
        pitch=layer
    )
    mask = np.any(layer != 0, axis=2)
    return layer, mask


def _blend_layers(prev_layer: np.ndarray, curr_layer: np.ndarray, alpha: float) -> np.ndarray:
    """Linear blend between two uint8 layers (HxWx3)."""
    return cv2.addWeighted(prev_layer, 1.0 - alpha, curr_layer, alpha, 0.0)


def _control_from_points(
    team0_xy: np.ndarray,
    team1_xy: np.ndarray,
    template: np.ndarray,
    step_px: int = CONTROL_STEP_PX
) -> tuple[float, float]:
    """
    Estimate pitch control from **points only** (no reliance on drawn pixels).

    We sample a coarse grid over the canonical pitch; each grid cell is assigned to the team
    whose nearest player is closest. Returns (p0, p1) ∈ [0..1] for team0 and team1.
    """
    # When both teams are empty, return neutral 50/50
    if (team0_xy is None or team0_xy.size == 0) and (team1_xy is None or team1_xy.size == 0):
        return 0.5, 0.5
    if team0_xy is None or team0_xy.size == 0:
        return 0.0, 1.0
    if team1_xy is None or team1_xy.size == 0:
        return 1.0, 0.0

    H, W = template.shape[:2]
    nx = max(1, W // step_px)
    ny = max(1, H // step_px)

    # Grid in canonical coordinates (same coordinate system as *_xy)
    xs = np.linspace(X_MIN, X_MAX, nx, dtype=np.float32)
    ys = np.linspace(Y_MIN, Y_MAX, ny, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)  # (N,2)

    # Squared distance to nearest player of each team
    g0 = grid[:, None, :] - team0_xy[None, :, :]
    d0 = np.min(np.einsum("ijk,ijk->ij", g0, g0), axis=1)

    g1 = grid[:, None, :] - team1_xy[None, :, :]
    d1 = np.min(np.einsum("ijk,ijk->ij", g1, g1), axis=1)

    assign0 = d0 <= d1
    p0 = float(np.count_nonzero(assign0)) / float(assign0.size)
    p1 = 1.0 - p0
    return p0, p1


def _draw_control_hud(img: np.ndarray, p0: float, p1: float) -> None:
    """
    Two-tone bar (team 0 left, team 1 right) + percentages.
    """
    h, w = img.shape[:2]
    bar_h = 22
    pad   = 12
    x0, y0 = pad, pad
    x1, y1 = w - pad, pad + bar_h
    W = x1 - x0

    # background rail
    cv2.rectangle(img, (x0, y0), (x1, y1), (40, 40, 40), -1, cv2.LINE_AA)

    w0 = int(W * np.clip(p0, 0.0, 1.0))
    c0 = tuple(sv.Color.from_hex(TEAM0_HEX).as_bgr())
    c1 = tuple(sv.Color.from_hex(TEAM1_HEX).as_bgr())
    cv2.rectangle(img, (x0, y0), (x0 + w0, y1), c0, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x0 + w0, y0), (x1, y1), c1, -1, cv2.LINE_AA)

    t0 = f"{int(round(p0 * 100))}% T0"
    t1 = f"{int(round(p1 * 100))}% T1"
    # left label (shadow + fill)
    cv2.putText(img, t0, (x0 + 8, y0 - 4 + bar_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, t0, (x0 + 8, y0 - 4 + bar_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    # right label
    sz1, _ = cv2.getTextSize(t1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(img, t1, (x1 - sz1[0] - 8, y0 - 4 + bar_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, t1, (x1 - sz1[0] - 8, y0 - 4 + bar_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


# ---------------- Main ----------------
def write_voronoi_2d_video(source_video: str, target_video: str) -> None:
    """
    Render a time-varying 2D Voronoi team control map over a canonical soccer pitch and save as video.
    Also overlays a **live pitch-control HUD** (percent area controlled by each team), computed
    robustly from player **points** (not pixels), with smooth interpolation between keyframes.

    Parameters
    ----------
    source_video : str
        Path to the input broadcast video (OpenCV-readable).
    target_video : str
        Path to the output video file. Resolution matches the library pitch template.

    Notes
    -----
    - Voronoi polygons are recomputed every `VORONOI_EVERY` frames and cross-faded in between.
    - Homography is refreshed every `KEYPOINT_EVERY` frames (min `MIN_KP` points).
    - The control HUD is derived by sampling a coarse canonical grid and assigning each cell
      to the nearest team (nearest player), avoiding anti-aliasing/paint artifacts.
    """
    rt = init_runtime(source_video, want_team_classifier=True)
    frames = sv.get_video_frames_generator(source_video)

    # Cross-fade state for polygons
    prev_layer: np.ndarray | None = None
    prev_mask:  np.ndarray | None = None
    curr_layer: np.ndarray | None = None
    curr_mask:  np.ndarray | None = None
    blend_t = 0
    blend_steps = max(1, VORONOI_EVERY)

    # Keyframe pitch-control percentages (team0, team1)
    prev_pct = (0.5, 0.5)
    curr_pct = (0.5, 0.5)

    with sv.VideoSink(target_video, video_info=rt.pitch_info) as sink:
        for i, frame in enumerate(tqdm(frames, total=rt.src_info.total_frames)):

            # 1) Detect & classify every frame so points are fresh
            ball, players, refs = detect_ball_and_players(frame, rt, conf_obj=OBJ_CONF)
            team_ids = classify_players(frame, players, rt.team_classifier, rt.team_id_map, frame_idx=i)

            # 2) Update homography periodically
            if (rt.vt is None) or (i % KEYPOINT_EVERY == 0):
                update_homography(frame, rt, keypoint_conf=0.30, min_points=MIN_KP, smooth_len=SMOOTH_H)

            # 3) Project anchors into pitch space if H is available
            if rt.vt is not None:
                pitch_ball = rt.vt.transform_points(anchors_bottom_center(ball))
                pitch_play = rt.vt.transform_points(anchors_bottom_center(players))
                pitch_refs = rt.vt.transform_points(anchors_bottom_center(refs))
            else:
                pitch_ball = np.empty((0, 2), np.float32)
                pitch_play = np.empty((0, 2), np.float32)
                pitch_refs = np.empty((0, 2), np.float32)

            # 4) Build/Blend Voronoi layers on keyframe cadence
            is_keyframe = (i % VORONOI_EVERY == 0)

            if is_keyframe:
                if rt.vt is None or pitch_play.size == 0:
                    new_layer = np.zeros_like(rt.template)
                    new_mask  = np.zeros(rt.template.shape[:2], dtype=bool)
                    new_pct   = curr_pct  # keep last % if we cannot compute new ones
                else:
                    t0 = pitch_play[team_ids == 0]
                    t1 = pitch_play[team_ids == 1]
                    new_layer, new_mask = _make_voronoi_layer(rt.template, t0, t1)
                    new_pct = _control_from_points(t0, t1, rt.template, step_px=CONTROL_STEP_PX)

                if curr_layer is None:
                    # First keyframe → seed prev/curr, skip initial fade
                    curr_layer, curr_mask = new_layer, new_mask
                    prev_layer, prev_mask = new_layer.copy(), new_mask.copy()
                    prev_pct = new_pct
                    curr_pct = new_pct
                    blend_t = blend_steps
                    voronoi_composited = curr_layer
                    active_mask = curr_mask
                    pct_now = curr_pct
                else:
                    # Start a new transition
                    prev_layer, prev_mask = curr_layer, curr_mask
                    curr_layer, curr_mask = new_layer, new_mask
                    prev_pct, curr_pct = curr_pct, new_pct
                    blend_t = 0
                    voronoi_composited = curr_layer
                    active_mask = curr_mask
                    pct_now = curr_pct
            else:
                # In-between frames: cross-fade polygons, interpolate percentages
                if prev_layer is None or curr_layer is None:
                    voronoi_composited = np.zeros_like(rt.template)
                    active_mask = np.zeros(rt.template.shape[:2], dtype=bool)
                    pct_now = curr_pct
                else:
                    blend_t = min(blend_t + 1, blend_steps)
                    a = blend_t / float(blend_steps)
                    voronoi_composited = _blend_layers(prev_layer, curr_layer, a)
                    active_mask = (prev_mask | curr_mask)
                    pct_now = ((1.0 - a) * prev_pct[0] + a * curr_pct[0],
                               (1.0 - a) * prev_pct[1] + a * curr_pct[1])

            # 5) Compose: pitch + blended Voronoi polygons (masked)
            canvas = rt.template.copy()
            if active_mask.any():
                bg = canvas[active_mask].astype(np.float32)
                fg = voronoi_composited[active_mask].astype(np.float32)
                blended = (1.0 - POLY_BLEND_ALPHA) * bg + POLY_BLEND_ALPHA * fg
                canvas[active_mask] = blended.astype(np.uint8)

            # 6) Live points (players/refs/ball)
            if pitch_play.size:
                if np.any(team_ids == 0):
                    canvas = draw_points_on_pitch(
                        CONFIG, pitch_play[team_ids == 0],
                        face_color=sv.Color.from_hex(TEAM0_HEX),
                        edge_color=sv.Color.BLACK,
                        radius=14, pitch=canvas
                    )
                if np.any(team_ids == 1):
                    canvas = draw_points_on_pitch(
                        CONFIG, pitch_play[team_ids == 1],
                        face_color=sv.Color.from_hex(TEAM1_HEX),
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

            # 7) HUD: live pitch-control bar (smoothly interpolated)
            _draw_control_hud(canvas, pct_now[0], pct_now[1])

            # 8) Output frame
            sink.write_frame(canvas)
