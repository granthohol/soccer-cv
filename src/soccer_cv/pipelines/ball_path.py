# src/soccer_cv/pipelines/path2d.py
from __future__ import annotations
from typing import Union, List
from tqdm import tqdm
import numpy as np
from collections import deque
import supervision as sv
from sports.annotators.soccer import draw_pitch, draw_paths_on_pitch, draw_points_on_pitch
from ..config import DEFAULT_CONFIG as CONFIG
from .common import init_runtime, detect_ball_and_players, update_homography, anchors_bottom_center

BALL_ID = 0
H_SMOOTH_MAXLEN = 5
DIST_THRESHOLD_PX = 1500.0

def _replace_outliers(positions: List[np.ndarray], thr: float) -> List[np.ndarray]:
    last: Union[np.ndarray, None] = None
    cleaned: List[np.ndarray] = []
    for p in positions:
        if p.size == 0:
            cleaned.append(p); continue
        if last is None:
            cleaned.append(p); last = p; continue
        if np.linalg.norm(p - last) > thr:
            cleaned.append(np.array([], dtype=np.float32))
        else:
            cleaned.append(p); last = p
    return cleaned


def write_ball_path_2d_video(
    source_video: str,
    target_video: str,
) -> None:
    
    rt = init_runtime(source_video, want_team_classifier=False)
    frames = sv.get_video_frames_generator(source_video)
    path_points: List[np.ndarray] = []

    with sv.VideoSink(target_video, video_info=rt.pitch_info) as sink:
        for i, frame in enumerate(tqdm(frames, total=rt.src_info.total_frames)):
            
            # detect ball only
            ball, players, _ = detect_ball_and_players(frame, rt, conf_obj=0.3)
            
            # H update
            if (rt.vt is None) or ( i% 5 == 0):
                update_homography(frame, rt, keypoint_conf=0.3, min_points=4, smooth_len=H_SMOOTH_MAXLEN)
                
            new_pt = np.array([], dtype=np.float32)
            if rt.vt is not None:
                ball_xy = anchors_bottom_center(ball)
                if ball_xy.size:
                    b_pitch = rt.vt.transform_points(ball_xy)
                    if b_pitch.size:
                        new_pt = b_pitch[0].astype(np.float32)
            
            # accumulate + clean
            path_points.append(new_pt)
            trail = np.asarray([p for p in _replace_outliers(path_points, thr=DIST_THRESHOLD_PX) if p.size == 2], dtype=np.float32)
            
            # draw and write 
            canvas = rt.template.copy()
            if trail.shape[0] >= 2:
                canvas = draw_paths_on_pitch(CONFIG, [trail], color=sv.Color.WHITE, pitch=canvas)
                
            if new_pt.size == 2:
                canvas = draw_points_on_pitch(CONFIG, new_pt.reshape(1, 2),
                                              face_color=sv.Color.WHITE, edge_color=sv.Color.BLACK,
                                              radius=10, pitch=canvas)
            sink.write_frame(canvas)