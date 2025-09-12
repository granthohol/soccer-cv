# src/soccer_cv/pipelines/possession.py
from __future__ import annotations
from collections import deque
from typing import Optional, Tuple

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
from ..utils import nearest_to_ball

# Tunables
KEYPOINT_EVERY   = 5          # refresh homography every K frames
OBJ_CONF         = 0.15       # object detection confidence threshold
SMOOTH_H         = 5          # homography smoothing window
MIN_KP           = 4          # minimum keypoints required for a homography
POS_RADIUS_PX    = 100.0       # max pitch distance ball->player to count possession
ROLLING_SECONDS  = 3.0        # rolling-average window length (seconds)
TEAM1_HEX        = '00BFFF'   # team 0 color (blue-ish)
TEAM2_HEX        = 'FF1493'   # team 1 color (pink)

BALL_ID = 0

def _draw_possession_bar(
    canvas: np.ndarray,
    pct_team0: float,
    pct_team1: float,
    *,
    height: int = 28,
    margin: int = 10,
    team0_hex: str = TEAM1_HEX,
    team1_hex: str = TEAM2_HEX,
    alpha: float = 0.9,
) -> np.ndarray:
    """Draw a compact stacked possession bar at the top of 'canvas'"""
    h, w = canvas.shape[:2]
    y0, y1 = margin, margin + height
    # Background strip (semi-transparent dark)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (margin, y0), (w - margin, y1), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, alpha * 0.4, canvas, 1 - alpha * 0.4, 0, dst=canvas)

    # Segment widths
    w0 = int((pct_team0 / 100.0) * (w - 2 * margin))
    w1 = (w - 2 * margin) - w0

    # Colors BGR for OpenCV
    c0 = sv.Color.from_hex(team0_hex).as_bgr()
    c1 = sv.Color.from_hex(team1_hex).as_bgr()

    # Draw segments
    x_start = margin
    cv2.rectangle(canvas, (x_start, y0), (x_start + w0, y1), c0, thickness=-1)
    cv2.rectangle(canvas, (x_start + w0, y0), (w - margin, y1), c1, thickness=-1)

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thick = 2
    txt0 = f"{pct_team0:4.1f}%"
    txt1 = f"{pct_team1:4.1f}%"
    # Shadow text for readability
    cv2.putText(canvas, txt0, (margin + 6, y1 + 22), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(canvas, txt0, (margin + 6, y1 + 22), font, scale, c0, thick, cv2.LINE_AA)

    sz1, _ = cv2.getTextSize(txt1, font, scale, thick)
    cv2.putText(canvas, txt1, (w - margin - sz1[0], y1 + 22), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(canvas, txt1, (w - margin - sz1[0], y1 + 22), font, scale, c1, thick, cv2.LINE_AA)

    return canvas    

def _nearest_possessor(ball_xy, players_xy, team_ids, radius_px) -> int:
    res = nearest_to_ball(
        ball_xy, players_xy,
        team_ids=team_ids,
        tracker_ids=None,
        radius_px=radius_px,
    )
    return res.team if (res.has_match and res.team is not None) else -1

def write_possession_2d_video(source_video: str, target_video: str) -> Tuple[float, float]:
    """
    Estimate rolling ball possession (%) per team and render it over a canonical 2D pitch video.

    The pipeline detects ball and players per frame, classifies players into two teams,
    periodically estimates a frame→pitch homography, projects the ball/player anchors to
    pitch coordinates, and assigns per-frame "possession" to the team whose nearest player
    lies within a configurable pitch distance of the ball. A rolling window (in seconds)
    converts those framewise assignments into percentages, drawn as a stacked bar and
    printed as text each frame. Final aggregate possession is printed to console when done.

    Parameters
    ----------
    source_video : str
        Path to input broadcast video.
    target_video : str
        Path to output MP4 (pitch-sized).
        
    Returns
    -------
    tuple(float, float) denoting the % possession for team 0 and team 1
    """
    
    # Initialize runtime (models, device, pitch canvas/video info, tracker, team classifier)
    rt = init_runtime(source_video, want_team_classifier=True)
    if not hasattr(rt, "team_id_map"):
        rt.team_id_map = {}  # ensure the classify_players cache exists
        
    frames = sv.get_video_frames_generator(source_video)
    fps = max(1.0, float(rt.src_info.fps))
    window_len = max(1, int(round(ROLLING_SECONDS * fps)))  # frames in rolling window
    
    # rolling possession: values in {0, 1, -1}
    poss_window: deque[int] = deque(maxlen=window_len)        
    
    # cumulative tallies
    total_known = 0
    total_team0 = 0
    total_team1 = 0
    
    last_possesor: Optional[int] = None
    
    template = draw_pitch(CONFIG)
    
    with sv.VideoSink(target_video, video_info=rt.pitch_info) as sink:
        for i, frame in enumerate(tqdm(frames, total=rt.src_info.total_frames)):
            
            # detect + track + team classify
            ball, players, refs = detect_ball_and_players(frame, rt, conf_obj=OBJ_CONF)
            team_ids = classify_players(frame, players, rt.team_classifier, rt.team_id_map, frame_idx=i)
            
            # homography refresh
            if (rt.vt is None) or (i % KEYPOINT_EVERY == 0):
                update_homography(frame, rt, keypoint_conf=0.3, min_points=MIN_KP, smooth_len=SMOOTH_H)
                
            # project to pitch coordinates
            if rt.vt is not None:
                pitch_ball = rt.vt.transform_points(anchors_bottom_center(ball))
                pitch_play = rt.vt.transform_points(anchors_bottom_center(players))
            else:
                pitch_ball = np.empty((0, 2), np.float32)
                pitch_play = np.empty((0, 2), np.float32)
                
            # frame possession decision (raw)
            raw_possession = _nearest_possessor(pitch_ball, pitch_play, team_ids, radius_px=POS_RADIUS_PX)
            if raw_possession in (0, 1):
                effective = raw_possession      # update memory
                last_possesor = raw_possession
            elif last_possesor is not None:
                effective = last_possesor       # carry forward last known team
            else:
                effective = -1                  # still unknown; should only be at start
                
            poss_window.append(effective)
            
            if effective in (0, 1):
                total_known += 1
                if effective == 0:
                    total_team0 += 1
                else:
                    total_team1 += 1
                    
            # rolling percentages from effective labels
            denom = max(1, total_known)        # avoid divide by zero early in the clip
            if total_known:
                pct0 = 100.0 * (total_team0 / denom)
                pct1 = 100.0 * (total_team1 / denom)
            else:
                pct0 = pct1 = 50.0           
                
            # draw 2D pitch + dots + possession bar
            canvas = template.copy()
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
            if pitch_ball.size:
                canvas = draw_points_on_pitch(
                    CONFIG, pitch_ball,
                    face_color=sv.Color.WHITE, edge_color=sv.Color.BLACK,
                    radius=10, pitch=canvas
                )

            canvas = _draw_possession_bar(canvas, pct_team0=pct0, pct_team1=pct1)
            sink.write_frame(canvas)    
            
        # final totals
        if total_known:
            final0 = 100.0 * (total_team0 / total_known)
            final1 = 100.0 * (total_team1 / total_known)
        else:
            final0 = final1 = 50.0
        
        print(
            f"[soccer-cv] Final possession — "
            f"Team 0: {final0:.1f}% | Team 1: {final1:.1f}% | "
            f"Unknown frames (pre-first-possession): {rt.src_info.total_frames - total_known}"            
        )
        
        return (total_team0 / total_known, total_team1 / total_known) if total_known else (0.5, 0.5)