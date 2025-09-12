# src/soccer_cv/pipelines/_common.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from collections import deque

import numpy as np
import supervision as sv
import cv2

# Import *light* stuff at module level; heavy deps inside functions
from ..devices import pick_device
from ..geometry import ViewTransformer
from ..models import load_default_object_model, load_default_pitch_model
from ..config import DEFAULT_CONFIG as CONFIG

BALL_ID, GK_ID, PLAYER_ID, REF_ID = 0, 1, 2, 3

@dataclass
class Runtime:
    """
    Shared state that persists for the whole pipeline to run.
    Keeps models, tracker, pitch canvas info, and homography smoothing buffer.
    """
    device: str
    object_model: object
    pitch_model: object
    template: np.ndarray        # pitch canvas (HxWx3)
    src_info: sv.VideoInfo      # original video info (input)
    pitch_info: sv.VideoInfo    # output video info (pitch-size)
    tracker: sv.ByteTrack       # multi object tracker that assigns stable tracker_id's to players/refs across frames
    team_classifier: Optional[object] = None
    team_id_map: dict[int, int] = field(default_factory=dict)   # track_id -> team_id
    
    # Homography smoothing
    H_buf: list[np.ndarray] = field(default_factory=list)   # rolling buffer of homography matrices; used to average/smooth the matrices
    vt: Optional[ViewTransformer] = None                    
    
def init_runtime(
        source_video: str,
        want_team_classifier: bool = False,
) -> Runtime:
    """
    1. Picks device (cpu/cuda/mps)
    2. Loads object and pitch models
    3. Fuse models if supported
    4. Builds a pitch canvas and pitch sized VideoInfo
    5. Prepares a ByteTrack tracker
    6. (Optional) Fits a TeamClassifier from early crops
    """
    
    try:
        from sports.annotators.soccer import (
            draw_pitch,
    )   
    except Exception as e:
        raise ImportError(
        "The 'sports' package is required for this feature. "
        "Install it separately:\n\n"
        "  pip install \"sports @ git+https://github.com/roboflow/sports.git@main\"\n"
    ) from e

    device = pick_device()

    # Lazy-import torch-heavy libs inside the function to keep top-level import fast
    try:
        from ultralytics import YOLO  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Ultralytics / PyTorch not installed.\n"
            "CPU: pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1\n"
            "GPU (CUDA 12.1): pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1 torchvision==0.19.1"
        ) from e

    obj_model: YOLO = load_default_object_model()
    pitch_model: YOLO = load_default_pitch_model()
    
    # fuse for slight inference time speedup
    for m in (obj_model, pitch_model):
        try:
            m.fuse()
        except Exception:
            pass

    src_info = sv.VideoInfo.from_video_path(source_video)   # read fps, resolution, count from input path
    # create blank pitch canvas; standardize output video size
    template = draw_pitch(CONFIG)   
    h, w = template.shape[:2]
    pitch_info = sv.VideoInfo(fps=src_info.fps, width=w, height=h, total_frames=src_info.total_frames) 

    tracker = sv.ByteTrack()
    tracker.reset()

    team_clf = None
    if want_team_classifier:
        # Fit a tiny color based classifier from early crops        
        try:
            from sports.common.team import (
                TeamClassifier,
            )
        except Exception as e:
            raise ImportError(
                "The 'sports' package is required for this feature. "
                "Install it separately:\n\n"
                "  pip install \"sports @ git+https://github.com/roboflow/sports.git@main\"\n"
            ) from e
        
        from ..utils import extract_crops

        crops = extract_crops(source_video)
        team_clf = TeamClassifier(device=device)
        team_clf.fit(crops if len(crops) else [np.zeros((32, 32, 3), dtype=np.uint8)])

    return Runtime(
        device=device,
        object_model=obj_model,
        pitch_model=pitch_model,
        template=template,
        src_info=src_info,
        pitch_info=pitch_info,
        tracker=tracker,
        team_classifier=team_clf,
    )

def detect_ball_and_players(
        frame: np.ndarray,
        rt: Runtime,
        conf_obj: float = 0.3,
) -> tuple[sv.Detections, sv.Detections, sv.Detections]:
    """
    Runs the object detector and tracker on a frame and returns:
    ball_dets, players_dets, refs_dets
    """
    # get all detections
    det_rets = rt.object_model.predict(frame, conf=conf_obj, verbose=False, device=rt.device)[0]
    dets = sv.Detections.from_ultralytics(det_rets)

    ball = dets[dets.class_id == BALL_ID]
    if ball.xyxy.size:
        ball.xyxy = sv.pad_boxes(ball.xyxy, px=10)  # pad the bounding box for ball detections

    
    others = dets[dets.class_id != BALL_ID].with_nms(threshold=0.5, class_agnostic=True)
    tracked = rt.tracker.update_with_detections(others)     # feed the others detections through the tracker to update tracker_ids

    players = tracked[tracked.class_id == PLAYER_ID]
    refs    = tracked[tracked.class_id == REF_ID]
    return ball, players, refs  
    

def classify_players(
        frame: np.ndarray,          # the current RGB/BGR frame (H,W,3), dtype=uint8
        players: sv.Detections,     # Supervision Detections for *players* in this frame
        team_clf: Optional[object], # a trained TeamClassifier 
        team_id_map: dict[int, int],   # rf.team_id_map
        *,
        frame_idx: int,             # current frame number
        refresh_stride: int = 60,   # fully refresh map every refresh_stride frames; default 60 is every second in 60fps
) -> np.ndarray:
    """
    Used to classify player detections by team
    
    Simple 'track_id -> team_id' mapping:
      - On frame 0 or every `refresh_stride` frames: classify ALL visible tracks and refresh the map.
      - On other frames: only classify tracks missing from the map (newly appeared).
      - Assigns players.class_id from the map. Unseen -> -1.

    Returns: players.class_id (np.ndarray[int]).
    """
    
    # if no player detections, return empty label array
    n = len(players)
    if n == 0:
        players.class_id = np.empty((0,), dtype=int)
        return players.class_id
    
    tracker_ids = getattr(players, "tracker_id", None)
    boxes = players.xyxy
    
    # decide if full refresh frame
    do_full_refresh = (frame_idx == 0) or (refresh_stride > 0 and (frame_idx % refresh_stride == 0))
    
    if do_full_refresh:
        # classify ALL visible tracks
        crops = [sv.crop_image(frame, box) for box in boxes]
        preds = team_clf.predict(crops).astype(int)
        for tracker_id, pred in zip(tracker_ids, preds):
            if tracker_id is not None:
                team_id_map[int(tracker_id)] = int(pred)    
    else:
        # classify ONLY tracks missing from the map (new arrivals)
        # get player indexes that need a refresh
        need_idx = [i for i, tracker_id in enumerate(tracker_ids) if (tracker_id is None) or (int(tracker_id) not in team_id_map)]
        if need_idx:
            crops = [sv.crop_image(frame, boxes[i]) for i in need_idx]
            preds = team_clf.predict(crops).astype(int)
            for j, i in enumerate(need_idx):
                tracker_id = tracker_ids[i]
                if tracker_id is not None:
                    team_id_map[int(tracker_id)] = int(preds[j])
    
    # Assign team_ids to detections
    assigned = np.array(    # get ream_ids
        [team_id_map.get(int(tracker_id), -1) if tracker_id is not None else -1 for tracker_id in tracker_ids],
        dtype=int
    )
    players.class_id = assigned
    return players.class_id
    
    

def update_homography(
    frame: np.ndarray,
    rt: Runtime,    
    keypoint_conf: float = 0.4,     # conf thresh for pitch landmarks
    min_points: int = 6,            # homography needs >=4 correspondences
    smooth_len: int = 7,            # number of recent homographies to average for stability
) -> bool:
    """
    Estimates a new homography matrix transformation (frame -> pitch) if enough keypoints exist,
    smooths it with a rolling average of the last `smooth_len` matrices,
    and stores it in rt.vt. Returns True if updated.
    
    This function:
    1. Detects pitch keypoints in the current frame
    2. Uses the detected points to compute a homography to the canonical pitch points
    3. Smooths the homography over the last few frames to reduce jitter
    4. Stores the smoothed transformer (rt.vt) for downstream projection calls
    """
    # run pitch keypoint model on this frame
    kp_res = rt.pitch_model.predict(frame, conf=keypoint_conf, verbose=False, device=rt.device)[0]
    kp = sv.KeyPoints.from_ultralytics(kp_res)
    
    # ensure feasibility
    if kp.xy.shape[0] == 0:
        return False
    m = kp.confidence[0] > 0.5
    if np.sum(m) < min_points:
        return False


    frame_ref = kp.xy[0][m]                     # observed landmark (x,y) positionsin the current camera frame
    pitch_ref = np.array(CONFIG.vertices)[m]    # canonical (x, y) positions of the corresponding landmarks in 2D pitch layout

    vt_new = ViewTransformer(source=frame_ref, target=pitch_ref)
    if getattr(vt_new, "matrix", None) is None:
        return False

    # Smooth the homography
    rt.H_buf.append(vt_new.matrix)          # add new homography matrix
    if len(rt.H_buf) > smooth_len:
        rt.H_buf = rt.H_buf[-smooth_len:]   # keep only the last `smooth_len` matrices
    H = np.mean(np.stack(rt.H_buf, axis=0), axis=0) # get the mean of the matrices
    if H[2, 2] != 0:
        H = H / H[2, 2]     # normalize
    vt_new.matrix = H
    rt.vt = vt_new
    return True


def anchors_bottom_center(dets: sv.Detections) -> np.ndarray:
    """Convenience for bottom-center anchor selection."""
    return dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) if dets.xyxy.size else np.empty((0, 2), np.float32)