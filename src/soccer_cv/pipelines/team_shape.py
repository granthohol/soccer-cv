# src/soccer_cv/pipelines/team_shape.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

from ..config import DEFAULT_CONFIG as CONFIG
from .common import (
    init_runtime,
    detect_ball_and_players,
    classify_players,
    update_homography,
    anchors_bottom_center,
)

try:
    from sports.annotators.soccer import (
        draw_points_on_pitch,
    )
except Exception as e:
    raise ImportError(
        "The 'sports' package is required for this feature. "
        "Install it separately:\n\n"
        "  pip install \"sports @ git+https://github.com/roboflow/sports.git@main\"\n"
    ) from e

# ---------- Tunables ----------
SHAPE_EVERY      = 5      # compute a new hull layer every K frames
KEYPOINT_EVERY   = 5      # refresh homography every K' frames
MIN_KP           = 4
OBJ_CONF         = 0.15
SMOOTH_H         = 5

TEAM0_HEX        = "00BFFF"   # blue
TEAM1_HEX        = "FF1493"   # pink

SHAPE_BLEND_A    = 0.45       # how strongly to paste (blended) shape layer onto the pitch
OUTLINE_THICK    = 3          # polygon outline thickness
FILL_OPACITY     = 1.0        # draw solid color on the shape layer (we’ll blend later)
HUD_COLOR        = (255, 255, 255)  # white text for metrics
HUD_SHADOW       = (0, 0, 0)        # black outline for readability
HUD_SCALE        = 0.6
HUD_THICK        = 2
HUD_LINE         = 18

# ---------- Small helpers ----------

_VERTS  = np.asarray(CONFIG.vertices, dtype=np.float32)
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


def _bgr_from_hex(hx: str) -> Tuple[int,int,int]:
    return tuple(sv.Color.from_hex(hx).as_bgr())

def _convex_hull(points: np.ndarray) -> Optional[np.ndarray]:
    """
    Returns Nx2 float32 convex hull points in canonical pitch space,
    or None if <3 points.
    """
    if points is None or points.size == 0 or len(points) < 3:
        return None
    # OpenCV expects (N,1,2) or (N,2); returns (M,1,2) in int or float
    hull = cv2.convexHull(points.astype(np.float32).reshape(-1, 1, 2))
    hull = hull.reshape(-1, 2).astype(np.float32)
    return hull

@dataclass
class ShapeMetrics:
    area: float
    width: float
    depth: float
    centroid: Tuple[float, float]

def _shape_metrics_from_points(points: np.ndarray, hull: Optional[np.ndarray]) -> ShapeMetrics:
    """
    Compute instantaneous metrics in canonical coordinates (pixels):
      - area: polygon area of hull (0 if none)
      - width: max_x - min_x across team points
      - depth: max_y - min_y across team points
      - centroid: mean of team points (fallback to hull centroid if needed)
    """
    if points is None or points.size == 0:
        return ShapeMetrics(area=0.0, width=0.0, depth=0.0, centroid=(0.0, 0.0))

    xs, ys = points[:, 0], points[:, 1]
    width  = float(xs.max() - xs.min()) if len(points) else 0.0
    depth  = float(ys.max() - ys.min()) if len(points) else 0.0

    if hull is not None and len(hull) >= 3:
        area = float(cv2.contourArea(hull.astype(np.float32)))
    else:
        area = 0.0

    cx = float(xs.mean())
    cy = float(ys.mean())
    
    area = round(area / 10000.0, 4)
    width = round(width / 100.0, 4)
    depth = round(depth / 100.0, 4)

    return ShapeMetrics(area=area, width=width, depth=depth, centroid=(cx, cy))

def _team_poly_layer(
    template: np.ndarray,
    hull_canon: np.ndarray,
    color_hex: str,
    alpha_fill: float = 0.6,
    outline_thick: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    h, w = template.shape[:2]
    rgb  = np.zeros((h, w, 3), dtype=np.float32)
    a    = np.zeros((h, w), dtype=np.float32)

    if hull_canon is None or len(hull_canon) < 3:
        return rgb, a

    # points for fillPoly / polylines
    pts_img = _canon_to_img_xy(hull_canon, template).astype(np.int32)
    # For fillPoly: list of polygons (M,2) -> (1,M,2)
    poly_list = [pts_img]
    # For polylines: contour shape should be (M,1,2) or a list of (M,2)
    contour = pts_img.reshape(-1, 1, 2)

    # alpha mask for the filled hull
    mask_u8 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_u8, poly_list, 255)
    a = (mask_u8.astype(np.float32) / 255.0) * float(alpha_fill)

    # constant team color (as Python tuple, not numpy array)
    color_bgr_tuple = tuple(int(c) for c in sv.Color.from_hex(color_hex).as_bgr())
    color_vec = np.array(color_bgr_tuple, dtype=np.float32)

    # premultiply color by alpha for later compositing
    rgb = (a[..., None] * color_vec[None, None, :])

    # Optional outline (draw directly on rgb; OpenCV accepts float32 images too)
    if outline_thick > 0:
        cv2.polylines(
            rgb, [contour], isClosed=True,
            color=color_bgr_tuple, thickness=outline_thick,
            lineType=cv2.LINE_AA
        )

    return rgb, a


def _blend_team_layers(
    rgb0: np.ndarray, a0: np.ndarray,
    rgb1: np.ndarray, a1: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Symmetric color mix in overlaps:
      color_out = (rgb0 + rgb1) / (a0 + a1)   where (a0+a1) > 0
      alpha_out = 1 - (1 - a0)*(1 - a1)       (union of opacities)
    Returns (rgb_out_uint8, alpha_out_float32[0..1])
    """
    eps = 1e-6
    sum_a = a0 + a1
    rgb_out = np.zeros_like(rgb0, dtype=np.float32)

    # normalize colors by total weight so brightness stays stable in overlap
    mask = sum_a > eps
    rgb_out[mask] = (rgb0[mask] + rgb1[mask]) / sum_a[mask, None]

    alpha_out = 1.0 - (1.0 - a0) * (1.0 - a1)
    return np.clip(rgb_out, 0, 255).astype(np.uint8), np.clip(alpha_out, 0.0, 1.0)


def _make_shape_layer(
    template: np.ndarray,
    team0_pts_canon: np.ndarray,
    team1_pts_canon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Returns:
      - blended RGB layer (HxWx3 uint8) of team shapes (with color mixing in overlap)
      - alpha map (HxW float32 0..1)
      - metrics dict as before
    """
    layer_h, layer_w = template.shape[:2]
    metrics = {"team0": ShapeMetrics(0,0,0,(0,0)), "team1": ShapeMetrics(0,0,0,(0,0))}

    # compute hulls & metrics in canonical space
    hull0 = _convex_hull(team0_pts_canon) if team0_pts_canon is not None and team0_pts_canon.size else None
    hull1 = _convex_hull(team1_pts_canon) if team1_pts_canon is not None and team1_pts_canon.size else None
    metrics["team0"] = _shape_metrics_from_points(team0_pts_canon, hull0) if hull0 is not None else metrics["team0"]
    metrics["team1"] = _shape_metrics_from_points(team1_pts_canon, hull1) if hull1 is not None else metrics["team1"]

    # per-team layers
    rgb0, a0 = _team_poly_layer(template, hull0, TEAM0_HEX, alpha_fill=0.55)
    rgb1, a1 = _team_poly_layer(template, hull1, TEAM1_HEX, alpha_fill=0.55)

    # symmetric blend → purple in overlap
    rgb_blend, a_blend = _blend_team_layers(rgb0, a0, rgb1, a1)

    return rgb_blend, a_blend, metrics


def _blend_layers(prev_layer: np.ndarray, curr_layer: np.ndarray, alpha: float) -> np.ndarray:
    """Linear blend between two uint8 layers (HxWx3)."""
    return cv2.addWeighted(prev_layer, 1.0 - alpha, curr_layer, alpha, 0.0)

def _draw_hud_metrics(
    canvas: np.ndarray,
    m0: ShapeMetrics,
    m1: ShapeMetrics,
) -> None:
    """
    Draw per-team metrics text (area/width/depth) and centroids as small dots.
    Left = Team 0, Right = Team 1.
    """
    h, w = canvas.shape[:2]
    # Positions for text blocks
    left_org  = (10, 22)
    right_org = (w - 320, 22)  # tweak width if text wraps

    def _put(block_origin, label):
        x, y = block_origin
        for i, line in enumerate(label.split("\n")):
            pt = (x, y + i * HUD_LINE)
            cv2.putText(canvas, line, pt, cv2.FONT_HERSHEY_SIMPLEX, HUD_SCALE, HUD_SHADOW, HUD_THICK+2, cv2.LINE_AA)
            cv2.putText(canvas, line, pt, cv2.FONT_HERSHEY_SIMPLEX, HUD_SCALE, HUD_COLOR, HUD_THICK, cv2.LINE_AA)

    # Format numbers (pixels)
    s0 = f"Team 0\nArea: {m0.area:.0f}\nWidth: {m0.width:.0f}\nDepth: {m0.depth:.0f}"
    s1 = f"Team 1\nArea: {m1.area:.0f}\nWidth: {m1.width:.0f}\nDepth: {m1.depth:.0f}"

    _put(left_org,  s0)
    _put(right_org, s1)

    # Draw centroid markers (white)
    for (cx, cy) in [m0.centroid, m1.centroid]:
        if cx > 0 or cy > 0:
            cv2.circle(canvas, (int(cx), int(cy)), 6, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (int(cx), int(cy)), 6, (0, 0, 0), 2, lineType=cv2.LINE_AA)  # outline

# ---------- Main pipeline ----------

def write_team_shape_video(
    source_video: str,
    target_video: str,
    *,
    shape_every: int = SHAPE_EVERY,
) -> None:
    """
    Render a continuous **team shape** overlay video on the canonical 2D pitch.

    The pipeline:
      • Detect → Track → Classify players each frame
      • Update homography every KEYPOINT_EVERY frames
      • Every `shape_every` frames: compute each team’s **convex hull** in pitch space,
        rasterize it as a color layer (black elsewhere), and **cross-fade** between the
        previous and current hull layers over the in-between frames.
      • Draw players/ball fresh every frame (no blending) and overlay **live metrics**:
        area (px²), width/depth (px), centroid.

    Parameters
    ----------
    source_video : str
        Input broadcast video path.
    target_video : str
        Output MP4 path (pitch-sized).
    shape_every : int, optional
        Recompute the hull layer every K frames. In-between frames cross-fade the layer.
        Default: 5.
    """
    max_frames = int(os.getenv("SOCCER_CV_MAX_FRAMES", "0")) or None

    # Bootstrap (models, device, tracker, template, pitch sizing)
    rt = init_runtime(source_video, want_team_classifier=True)
    if not hasattr(rt, "team_id_map"):
        rt.team_id_map = {}

    frames = sv.get_video_frames_generator(source_video)

    # Cross-fade state
    prev_rgb, prev_a = None, None      # previous keyframe's RGB (HxWx3 uint8) and ALPHA (HxW float32 0..1)
    curr_rgb, curr_a = None, None      # current keyframe's RGB and ALPHA
    # Metrics for previous/current keyframe (for smooth interpolation)
    prev_metrics = {"team0": ShapeMetrics(0,0,0,(0,0)), "team1": ShapeMetrics(0,0,0,(0,0))}
    curr_metrics = {"team0": ShapeMetrics(0,0,0,(0,0)), "team1": ShapeMetrics(0,0,0,(0,0))}
    blend_t    = 0
    blend_steps = max(1, shape_every)


    with sv.VideoSink(target_video, video_info=rt.pitch_info) as sink:
        for i, frame in enumerate(tqdm(frames, total=rt.src_info.total_frames)):
            if max_frames is not None and i >= max_frames:
                break

            # 1) Detect & classify per frame
            ball, players, refs = detect_ball_and_players(frame, rt, conf_obj=OBJ_CONF)
            team_ids = classify_players(frame, players, rt.team_classifier, rt.team_id_map, frame_idx=i)

            # 2) Update homography periodically
            if (rt.vt is None) or (i % KEYPOINT_EVERY == 0):
                update_homography(frame, rt, keypoint_conf=0.30, min_points=MIN_KP, smooth_len=SMOOTH_H)

            # 3) Project to canonical pitch
            if rt.vt is not None:
                pitch_play = rt.vt.transform_points(anchors_bottom_center(players))
                pitch_ball = rt.vt.transform_points(anchors_bottom_center(ball)) if ball.xyxy.size else np.empty((0,2), np.float32)
            else:
                pitch_play = np.empty((0, 2), np.float32)
                pitch_ball = np.empty((0, 2), np.float32)

            # Split players by team id (ignore unknown = -1)
            t0_pts = pitch_play[team_ids == 0] if pitch_play.size else np.empty((0,2), np.float32)
            t1_pts = pitch_play[team_ids == 1] if pitch_play.size else np.empty((0,2), np.float32)

            is_keyframe = (i % shape_every == 0)
            if is_keyframe:
                if rt.vt is None or (t0_pts.size == 0 and t1_pts.size == 0):
                    new_rgb  = np.zeros_like(rt.template, dtype=np.uint8)
                    new_a    = np.zeros(rt.template.shape[:2], dtype=np.float32)
                    new_metrics = {"team0": ShapeMetrics(0,0,0,(0,0)), "team1": ShapeMetrics(0,0,0,(0,0))}
                else:
                    new_rgb, new_a, new_metrics = _make_shape_layer(rt.template, t0_pts, t1_pts)

                if curr_rgb is None:
                    # first keyframe: seed both prev/curr and skip the initial fade
                    curr_rgb, curr_a = new_rgb, new_a
                    prev_rgb, prev_a = new_rgb.copy(), new_a.copy()
                    curr_metrics = new_metrics
                    prev_metrics = {"team0": curr_metrics["team0"], "team1": curr_metrics["team1"]}
                    blend_t = blend_steps
                    composited_rgb = curr_rgb
                    composited_a   = curr_a
                    m0, m1 = curr_metrics["team0"], curr_metrics["team1"]
                else:
                    # Start new transition
                    prev_rgb, prev_a = curr_rgb, curr_a
                    curr_rgb, curr_a = new_rgb, new_a
                    prev_metrics = curr_metrics
                    curr_metrics = new_metrics
                    blend_t = 0
                    composited_rgb = curr_rgb
                    composited_a   = curr_a
                    m0, m1 = curr_metrics["team0"], curr_metrics["team1"]

            else:
                # In-between: cross-fade BOTH color and alpha, then interpolate metrics
                if prev_rgb is None or curr_rgb is None:
                    composited_rgb = np.zeros_like(rt.template, dtype=np.uint8)
                    composited_a   = np.zeros(rt.template.shape[:2], dtype=np.float32)
                    m0 = prev_metrics["team0"]
                    m1 = prev_metrics["team1"]
                else:
                    blend_t = min(blend_t + 1, blend_steps)
                    t = blend_t / float(blend_steps)

                    # cross-fade color (uint8) and alpha (float)
                    composited_rgb = cv2.addWeighted(prev_rgb, 1.0 - t, curr_rgb, t, 0.0)
                    composited_a   = (1.0 - t) * prev_a + t * curr_a

                    # interpolate metrics for smooth HUD
                    def _lerp(a, b, u): return a + (b - a) * u
                    pm0, cm0 = prev_metrics["team0"], curr_metrics["team0"]
                    pm1, cm1 = prev_metrics["team1"], curr_metrics["team1"]
                    m0 = ShapeMetrics(
                        area=_lerp(pm0.area, cm0.area, t),
                        width=_lerp(pm0.width, cm0.width, t),
                        depth=_lerp(pm0.depth, cm0.depth, t),
                        centroid=(_lerp(pm0.centroid[0], cm0.centroid[0], t),
                                _lerp(pm0.centroid[1], cm0.centroid[1], t))
                    )
                    m1 = ShapeMetrics(
                        area=_lerp(pm1.area, cm1.area, t),
                        width=_lerp(pm1.width, cm1.width, t),
                        depth=_lerp(pm1.depth, cm1.depth, t),
                        centroid=(_lerp(pm1.centroid[0], cm1.centroid[0], t),
                                _lerp(pm1.centroid[1], cm1.centroid[1], t))
                    )


            # 4) Compose onto pitch using per-pixel alpha
            canvas = rt.template.copy().astype(np.float32)

            if 'composited_a' in locals() and np.any(composited_a > 0):
                # Optionally scale maximum opacity by SHAPE_BLEND_A
                a = (composited_a * float(SHAPE_BLEND_A))[..., None].astype(np.float32)  # HxWx1
                fg = composited_rgb.astype(np.float32)                                   # HxWx3
                canvas = canvas * (1.0 - a) + fg * a

            canvas = canvas.astype(np.uint8)


            # 5) Optional: draw current players/ball on top (fresh every frame)
            if pitch_play.size:
                if np.any(team_ids == 0):
                    canvas = draw_points_on_pitch(
                        config=CONFIG,
                        xy=t0_pts,
                        face_color=sv.Color.from_hex(TEAM0_HEX),
                        edge_color=sv.Color.BLACK,
                        radius=12,
                        pitch=canvas
                    )
                if np.any(team_ids == 1):
                    canvas = draw_points_on_pitch(
                        config=CONFIG,
                        xy=t1_pts,
                        face_color=sv.Color.from_hex(TEAM1_HEX),
                        edge_color=sv.Color.BLACK,
                        radius=12,
                        pitch=canvas
                    )
            if pitch_ball.size:
                canvas = draw_points_on_pitch(
                    config=CONFIG,
                    xy=pitch_ball,
                    face_color=sv.Color.WHITE,
                    edge_color=sv.Color.BLACK,
                    radius=10,
                    pitch=canvas
                )

            # 6) HUD metrics + centroid dots
            _draw_hud_metrics(canvas, m0, m1)

            # 7) Write frame
            sink.write_frame(canvas)
