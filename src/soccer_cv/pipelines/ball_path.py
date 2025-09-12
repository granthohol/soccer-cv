# src/soccer_cv/pipelines/path2d.py
from __future__ import annotations
from typing import Union, List
from tqdm import tqdm
import numpy as np
from collections import deque
import supervision as sv
from ..config import DEFAULT_CONFIG as CONFIG
from .common import init_runtime, detect_ball_and_players, update_homography, anchors_bottom_center

try:
    from sports.annotators.soccer import (
        draw_paths_on_pitch,
    )
except Exception as e:
    raise ImportError(
        "The 'sports' package is required for this feature. "
        "Install it separately:\n\n"
        "  pip install \"sports @ git+https://github.com/roboflow/sports.git@main\"\n"
    ) from e

BALL_ID = 0
H_SMOOTH_MAXLEN = 5             # how many recent homography matrices kept to smooth
DIST_THRESHOLD_PX = 1500.0      # in pitch pixels, the maximum allowed jump between consecutive ball positions before treating point as an outlier

def _replace_outliers(positions: List[np.ndarray], thr: float) -> List[np.ndarray]:
    last: Union[np.ndarray, None] = None
    cleaned: List[np.ndarray] = []
    for p in positions:
        if p.size == 0:
            cleaned.append(p); continue
        if last is None:
            cleaned.append(p); last = p; continue
        if np.linalg.norm(p - last) > thr:  # if euclidean distance to next point is larger than thresh, treat as outlier
            cleaned.append(np.array([], dtype=np.float32))
        else:
            cleaned.append(p); last = p
    return cleaned


def write_ball_path_2d_video(
    source_video: str,
    target_video: str,
) -> None:
    """
    Render the ball's trajectory as a 2D overlay on a canonical soccer pitch and write it as a video.

    This pipeline ingests a broadcast (camera) video, detects the ball in each frame using the
    default object-detection model, periodically estimates a frame→pitch homography from pitch
    keypoints, projects the ball position into pitch coordinates, and draws the evolving path
    on a pitch-sized canvas. Outlier jumps are suppressed with a simple distance-based filter
    to reduce spurious spikes when detections momentarily fail.

    Parameters
    ----------
    source_video : str
        Path to the input broadcast video. Must be readable by OpenCV/Supervision.
    target_video : str
        Path to the output MP4 (or other codec supported by Supervision). The output resolution
        matches the pitch template defined by the library's default configuration.
        
    Notes
    -----
    - Models, device selection (CPU/CUDA/MPS), pitch template, and video I/O are initialized via
      ``init_runtime``. If PyTorch/Ultralytics are not installed correctly, import/runtime errors will propagate.
      
    Returns
    -------
    None
        Writes ``target_video`` to disk; raises on I/O/model errors.

    Examples
    --------
    >>> from soccer_cv import write_ball_path_2d_video
    >>> write_ball_path_2d_video("content/match_clip.mp4", "output/ball_path.mp4")
    """
    
    rt = init_runtime(source_video, want_team_classifier=False)
    frames = sv.get_video_frames_generator(source_video)
    path_points: List[np.ndarray] = []      # one item per frame; either (2,) pitch point or empty

    with sv.VideoSink(target_video, video_info=rt.pitch_info) as sink:
        for i, frame in enumerate(tqdm(frames, total=rt.src_info.total_frames)):
            
            # detect ball only
            ball, _, _ = detect_ball_and_players(frame, rt)
            
            # homography update every 5 frames
            if (rt.vt is None) or ( i% 5 == 0):
                update_homography(frame, rt, keypoint_conf=0.3, min_points=4, smooth_len=H_SMOOTH_MAXLEN)
                
            # project ball into space    
            new_pt = np.array([], dtype=np.float32)
            if rt.vt is not None:
                ball_xy = anchors_bottom_center(ball)   # get bottom center anchor of balls box in frame coordinates
                if ball_xy.size:
                    # get best ball prediction by confidecne
                    conf = getattr(ball, "confidence", None)
                    idx = int(np.argmax(conf)) if (conf is not None and len(conf)) else 0
                    
                    sel_xy = ball_xy[idx:idx+1]
                    b_pitch = rt.vt.transform_points(sel_xy)   # transform to 2D
                    if b_pitch.size:
                        new_pt = b_pitch[0].astype(np.float32)
            
            # accumulate + clean the path
            path_points.append(new_pt)
            trail = np.asarray([p for p in _replace_outliers(path_points, thr=DIST_THRESHOLD_PX) if p.size == 2], dtype=np.float32)
            
            # draw and write 
            canvas = rt.template.copy()
            if trail.shape[0] >= 2:
                canvas = draw_paths_on_pitch(CONFIG, [trail], color=sv.Color.WHITE, pitch=canvas)

            sink.write_frame(canvas)