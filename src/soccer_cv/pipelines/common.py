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
    from sports.annotators.soccer import draw_pitch

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
        from sports.common.team import TeamClassifier
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
        frame: np.ndarray,
        players: sv.Detections,
        team_clf: Optional[object],
        team_map: dict[int, int],   # rf.team_id_map
        *,
        frame_idx: int,
        refresh_stride: int = 60,   # re run full refresh every N frames
) -> np.ndarray:
    """
    Simple 'track_id -> team_id' mapping:
      - On frame 0 or every `refresh_stride` frames: classify ALL visible tracks and refresh the map.
      - On other frames: only classify tracks missing from the map (newly appeared).
      - Assigns players.class_id from the map. Unseen -> -1.

    Returns: players.class_id (np.ndarray[int]).
    """
    n = len(players)
    if n == 0:
        players.class_id = np.empty((0,), dtype=int)
        return players.class_id
    
    task_ids = getattr(players, "tracker_id", None)
    boxes = players.xyxy
    
    do_full_refresh = (frame_idx == 0) or (refresh_stride > 0 and (frame_idx % refresh_stride == 0))
    
    if do_full_refresh:
        # classify ALL visible tracks
        crops = [sv.crop_image(frame, box) for box in boxes]
        preds = team_clf.predict(crops).astype(int)
        for task_id, p in zip(task_ids, preds):
            if task_id is not None:
                team_map[int(task_id)] = int(p)    
    else:
        # classify ONLY tracks missing from the map (new arrivals)
        need_idx = [i for i, tid in enumerate(task_ids) if (tid is None) or (int(tid) not in team_map)]
        if need_idx:
            crops = [sv.crop_image(frame, boxes[i]) for i in need_idx]
            preds = team_clf.predict(crops).astype(int)
            for j, i in enumerate(need_idx):
                tid = task_ids[i]
                if tid is not None:
                    team_map[int(tid)] = int(preds[j])
    
    # Assign from map (unknown = -1)
    assigned = np.array(
        [team_map.get(int(tid), -1) if tid is not None else -1 for tid in task_ids],
        dtype=int
    )
    players.class_id = assigned
    return players.class_id
    
    

def update_homography(
    frame: np.ndarray,
    rt: Runtime,
    keypoint_conf: float = 0.3,
    min_points: int = 4,
    smooth_len: int = 5,
) -> bool:
    """
    Estimates a new homography (frame -> pitch) if enough keypoints exist,
    smooths it with a rolling average of the last `smooth_len` matrices,
    and stores it in rt.vt. Returns True if updated.
    """
    kp_res = rt.pitch_model.predict(frame, conf=keypoint_conf, verbose=False, device=rt.device)[0]
    kp = sv.KeyPoints.from_ultralytics(kp_res)
    if kp.xy.shape[0] == 0:
        return False

    m = kp.confidence[0] > 0.5
    if np.sum(m) < min_points:
        return False

    frame_ref = kp.xy[0][m]
    pitch_ref = np.array(CONFIG.vertices)[m]

    vt_new = ViewTransformer(source=frame_ref, target=pitch_ref)
    if getattr(vt_new, "m", None) is None:
        return False

    # Smooth the homography
    rt.H_buf.append(vt_new.m)
    if len(rt.H_buf) > smooth_len:
        rt.H_buf = rt.H_buf[-smooth_len:]
    H = np.mean(np.stack(rt.H_buf, axis=0), axis=0)
    if H[2, 2] != 0:
        H = H / H[2, 2]
    vt_new.m = H
    rt.vt = vt_new
    return True


def anchors_bottom_center(dets: sv.Detections) -> np.ndarray:
    """Convenience for bottom-center anchor selection."""
    return dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) if dets.xyxy.size else np.empty((0, 2), np.float32)