# src/soccer_cv/pipelines/heatmaps.py
from __future__ import annotations
import os
from typing import Deque, Optional, List, Dict, Tuple
from collections import deque

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
import math

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


# --- VIDEO OVERLAY BY TEAM HEATMAPS ---

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
            
            

# ---- HEATMAP BY PLAYER ---

def _put_label_bgr(img: np.ndarray, text: str, org=(10, 28), color=(255,255,255)) -> None:
    """Draw a small bold label (OpenCV BGR)."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0,0,0), 3, cv2.LINE_AA)       # outline
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color,    2, cv2.LINE_AA)       # fill

MIN_PRESENCE_FRAMES = 8   # drop tracks with very few visible frames (noise)
TOP_K_PER_TEAM      = 10  # keep only top-k players per team


def _tile_panels(panels: List[np.ndarray], cols: int) -> np.ndarray:
    """Stack equally sized HxW panels row-major into a grid image."""
    assert panels, "no panels to tile"
    H, W = panels[0].shape[:2]
    n = len(panels)
    rows = math.ceil(n / max(1, cols))
    # pad with blanks to fill the grid
    if n < rows * cols:
        panels = panels + [np.zeros_like(panels[0]) for _ in range(rows * cols - n)]
    grid = []
    for r in range(rows):
        row_imgs = panels[r*cols:(r+1)*cols]
        grid.append(np.concatenate(row_imgs, axis=1))
    return np.concatenate(grid, axis=0)


def write_team_player_heatmap_grids(
    source_video: str,
    output_dir: str,
    *,
    grid_cols: int = 4,
    normalize: str = "presence",   # "presence" or "clip"
    cmap_name: str = "JET",
    alpha_max: float = 0.7,
    blur_sigma: float = 5.0,
) -> Tuple[str, str]:
    """
    Generate **per-player cumulative heatmaps** for each team and save them as two tiled PNG images.

    This function processes a broadcast soccer video once, tracks players, projects their
    locations into a canonical 2D pitch via homography, and accumulates each player's
    on-pitch presence into a heat grid. For each team, its players heatmaps are rendered as
    a grid of panels (each panel = one player, labeled with their `#track_id`) over the pitch.
    The two output files are:

    - `team0_heatmaps_grid.png` — Team 0’s players
    - `team1_heatmaps_grid.png` — Team 1’s players

    The heatmaps are **cumulative over the entire clip** (not rolling).

    Parameters
    ----------
    source_video : str
        Path to the input broadcast video.
    output_dir : str
        Directory where the two PNG files will be written. The directory is created if needed.
    grid_cols : int, optional
        Number of columns in the tiled grid for each team image. Rows are computed automatically
        based on the number of selected players (up to 10). Default is 4.
    normalize : {"presence", "clip"}, optional
        How to normalize heat intensity before rendering:
        - `"presence"`: divide each player's grid by **that player's** number of visible frames
          (fair for substitutes; emphasizes their personal usage).
        - `"clip"`: divide by **total output frames** processed (comparable across players; penalizes
          players who were on the pitch less time).
        Default is `"presence"`.
    cmap_name : str, optional
        OpenCV colormap name used to colorize the heat (e.g., `"JET"`, `"HOT"`, `"TURBO"`).
        Default is `"JET"`.
    alpha_max : float, optional
        Maximum alpha used when blending the heatmap over the pitch (0–1). Default is 0.7.
    blur_sigma : float, optional
        Gaussian sigma (in grid pixels) applied to the accumulated heat grid prior to rendering.
        Smooths noisy localization. Default is 5.0.

    Returns
    -------
    (str, str)
        Tuple of file paths `(team0_png, team1_png)` pointing to the written PNGs.

    Notes
    -----
    - The routine uses the library’s runtime bootstrap to load object and pitch keypoint models,
      run detection + tracking (ByteTrack), classify players into two teams, estimate per-frame
      homography, and project detections to canonical pitch coordinates.
    - Player selection per team is capped at **top 10** by “data volume” (sum of grid counts),
      with a minimum presence threshold to filter out brief blips and ID noise.
    - If homography cannot be estimated reliably for long stretches (e.g., too few pitch keypoints),
      little or no heat will accumulate; verify your pitch keypoint model and thresholds.
    - ID switches in tracking can split a real player’s heat across multiple `track_id`s; the function
      renders each `track_id` independently.
    - Color scaling uses percentile-based exposure for a final, readable snapshot, so low-count regions
      remain visible.

    Example
    -------
    >>> from soccer_cv.pipelines.heatmaps import write_team_player_heatmap_grids
    >>> out0, out1 = write_team_player_heatmap_grids(
    ...     "content/121364_0.mp4",
    ...     "outputs/heatmaps",
    ...     grid_cols=4,
    ...     normalize="presence",
    ...     cmap_name="JET",
    ...     alpha_max=0.7,
    ...     blur_sigma=5.0,
    ... )
    >>> print(out0, out1)
    outputs/heatmaps/team0_heatmaps_grid.png outputs/heatmaps/team1_heatmaps_grid.png

    Raises
    ------
    RuntimeError
        If required runtime dependencies (e.g., PyTorch/Ultralytics) are not installed or models
        cannot be loaded; errors are propagated from the runtime initialization and model calls.

    See Also
    --------
    write_team_heatmaps_video : Renders cumulative team heatmaps as a split-screen video.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Initialize runtime
    rt = init_runtime(source_video, want_team_classifier=True)
    if not hasattr(rt, "team_id_map"):
        rt.team_id_map = {}

    frames = sv.get_video_frames_generator(source_video)
    pitch_template = rt.template

    # 2) Per-team, per-player accumulators
    heats0: Dict[int, np.ndarray] = {}   # tid -> (GRID_H, GRID_W) float32 counts
    heats1: Dict[int, np.ndarray] = {}
    pres0: Dict[int, int] = {}           # tid -> presence frames
    pres1: Dict[int, int] = {}
    frames_seen = 0

    # 3) Iterate frames once
    for i, frame in enumerate(tqdm(frames, total=rt.src_info.total_frames)):
        # Detect + classify players
        _, players, _ = detect_ball_and_players(frame, rt, conf_obj=OBJ_CONF)
        team_ids = classify_players(frame, players, rt.team_classifier, rt.team_id_map, frame_idx=i)

        # Update homography periodically
        if (rt.vt is None) or (i % KEYPOINT_EVERY == 0):
            update_homography(frame, rt, keypoint_conf=0.30, min_points=MIN_KP, smooth_len=SMOOTH_H)

        frames_seen += 1

        # Accumulate only if valid homography and players present
        if rt.vt is not None and players.xyxy.size:
            anchors = anchors_bottom_center(players)       # (N,2) in frame
            pitch_xy = rt.vt.transform_points(anchors)     # (N,2) in pitch

            tids = getattr(players, "tracker_id", None)
            if tids is not None and len(tids) == len(pitch_xy):
                for k in range(len(tids)):
                    tid_raw = tids[k]
                    if tid_raw is None:
                        continue
                    tid = int(tid_raw)
                    tm = int(team_ids[k]) if k < len(team_ids) else -1
                    if tm not in (0, 1):
                        continue

                    heats = heats0 if tm == 0 else heats1
                    pres  = pres0  if tm == 0 else pres1

                    if tid not in heats:
                        heats[tid] = np.zeros((GRID_H, GRID_W), dtype=np.float32)
                        pres[tid]  = 0

                    xy = pitch_xy[k:k+1]
                    if xy.size:
                        _accumulate_heat(heats[tid], xy)
                        pres[tid] += 1

    # Helper: pick top-K players by “data volume” (sum of heat), tie-break by presence
    def _top_k_by_data(heats: Dict[int, np.ndarray], pres: Dict[int, int], k: int) -> List[int]:
        scored = []
        for tid, grid in heats.items():
            s = float(grid.sum())    # total visits across grid cells
            p = int(pres.get(tid, 0))
            if p < MIN_PRESENCE_FRAMES or s <= 0.0:
                continue
            scored.append((tid, s, p))
        # sort: most data first, then most presence
        scored.sort(key=lambda t: (t[1], t[2]), reverse=True)
        return [tid for (tid, _, _) in scored[:k]]

    top0 = _top_k_by_data(heats0, pres0, TOP_K_PER_TEAM)
    top1 = _top_k_by_data(heats1, pres1, TOP_K_PER_TEAM)

    # 4) Render a grid image for each team (only the selected top-K)
    def _render_team_grid(heats: Dict[int, np.ndarray], pres: Dict[int, int],
                          tids: List[int], label_team: int) -> np.ndarray:
        panels: List[np.ndarray] = []
        for tid in tids:
            grid = heats[tid]
            if normalize == "presence":
                denom = max(1, pres.get(tid, 0))
            else:
                denom = max(1, frames_seen)

            panel = _render_heat_overlay_colormap(
                base_pitch=pitch_template,
                grid=grid,
                frames_seen=denom,
                blur_sigma=blur_sigma,
                cmap_name=cmap_name,
                alpha_max=alpha_max,
                exposure_pct=98,   # expose once for the final snapshot → makes low counts visible
            )
            _put_label_bgr(panel, f"# {tid} (team {label_team})")
            panels.append(panel)

        if not panels:
            # produce a single blank pitch with a note if no valid players
            p = pitch_template.copy()
            _put_label_bgr(p, f"No valid players for team {label_team}", color=(0, 255, 255))
            panels = [p]

        return _tile_panels(panels, cols=grid_cols)

    grid0 = _render_team_grid(heats0, pres0, top0, label_team=0)
    grid1 = _render_team_grid(heats1, pres1, top1, label_team=1)

    # 5) Save PNGs
    path0 = os.path.join(output_dir, "team0_heatmaps_grid.png")
    path1 = os.path.join(output_dir, "team1_heatmaps_grid.png")
    cv2.imwrite(path0, grid0)
    cv2.imwrite(path1, grid1)

    return path0, path1