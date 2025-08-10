# src/soccer_cv/pipelines/path2d.py
from __future__ import annotations
import numpy as np
from collections import deque
from typing import List, Union
import supervision as sv
from tqdm import tqdm
from sports.annotators.soccer import draw_pitch, draw_paths_on_pitch, draw_points_on_pitch

from ..config import DEFAULT_CONFIG as CONFIG
from ..devices import pick_device
from ..models import load_default_player_model, load_default_pitch_model
from ..geometry import ViewTransformer

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
    
    try:
        from ultralytics import YOLO  # pulls in torch
    except Exception as e:
        raise RuntimeError(
            "Ultralytics / PyTorch not installed properly.\n"
            "CPU install: pip install torch==2.4.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html\n"
            "GPU install (CUDA 12.1):  pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1 torchvision==0.19.1"
        ) from e

    device = pick_device()
    OBJECT_DETECTION_MODEL = load_default_player_model()
    PITCH_DETECTION_MODEL  = load_default_pitch_model()

    video_info = sv.VideoInfo.from_video_path(source_video)
    frames = sv.get_video_frames_generator(source_video)

    template = draw_pitch(config=CONFIG)
    h, w = template.shape[:2]
    pitch_info = sv.VideoInfo(fps=video_info.fps, width=w, height=h, total_frames=video_info.total_frames)

    H_buf: deque[np.ndarray] = deque(maxlen=H_SMOOTH_MAXLEN)
    path_points: List[np.ndarray] = []

    with sv.VideoSink(target_video, video_info=pitch_info) as sink:
        for frame in tqdm(frames, total=video_info.total_frames):
            # 1) detect ball
            det_res = OBJECT_DETECTION_MODEL.predict(frame, conf=0.3, verbose=False, device=device)[0]
            dets = sv.Detections.from_ultralytics(det_res)
            ball = dets[dets.class_id == BALL_ID]

            ball_xy_frame = np.empty((0,2), np.float32)
            if ball.xyxy.size:
                anchors = ball.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                if anchors.size:
                    idx = int(np.argmax(getattr(ball, "confidence", np.array([1.0]*len(anchors)))))
                    ball_xy_frame = anchors[idx:idx+1]

            # 2) pitch keypoints & homography (frame -> pitch)
            new_pitch_point = None
            kp_res = PITCH_DETECTION_MODEL.predict(frame, conf=0.3, verbose=False, device=device)[0]
            kp = sv.KeyPoints.from_ultralytics(kp_res)
            if kp.xy.shape[0] and ball_xy_frame.size:
                m = kp.confidence[0] > 0.5
                if np.sum(m) >= 4:
                    frame_ref = kp.xy[0][m]
                    pitch_ref = np.array(CONFIG.vertices)[m]
                    vt = ViewTransformer(source=frame_ref, target=pitch_ref)
                    if vt.m is not None:
                        H_buf.append(vt.m)
                        H = np.mean(np.stack(H_buf, axis=0), axis=0)
                        if H[2,2] != 0: H = H / H[2,2]
                        vt.m = H
                        ball_xy_pitch = vt.transform_points(ball_xy_frame)
                        if ball_xy_pitch.size:
                            new_pitch_point = ball_xy_pitch[0].astype(np.float32)

            # 3) accumulate + clean
            path_points.append(new_pitch_point if new_pitch_point is not None
                               else np.array([], dtype=np.float32))
            cleaned = _replace_outliers(path_points, thr=DIST_THRESHOLD_PX)
            trail = np.asarray([p for p in cleaned if p.size == 2], dtype=np.float32)

            # 4) draw & write
            canvas = template.copy()
            if trail.shape[0] >= 2:
                canvas = draw_paths_on_pitch(CONFIG, [trail], color=sv.Color.WHITE, pitch=canvas)
            if cleaned[-1].size == 2:
                canvas = draw_points_on_pitch(CONFIG, cleaned[-1].reshape(1,2),
                                              face_color=sv.Color.WHITE, edge_color=sv.Color.BLACK,
                                              radius=10, pitch=canvas)
            sink.write_frame(canvas)