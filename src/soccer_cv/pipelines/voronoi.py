# src/soccer_cv/pipelines/voronoi2d.py
from __future__ import annotations
import numpy as np
import supervision as sv
from tqdm import tqdm
from sports.common.team import TeamClassifier
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch, draw_pitch_voronoi_diagram

from ..config import DEFAULT_CONFIG as CONFIG
from ..devices import pick_device
from ..models import load_default_object_model, load_default_pitch_model
from ..geometry import ViewTransformer
from ..utils import extract_crops, resolve_goalkeepers_team_id

BALL_ID, GK_ID, PLAYER_ID, REF_ID = 0, 1, 2, 3

def write_voronoi_2d_video(
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
    OBJECT_DETECTION_MODEL = load_default_object_model()
    PITCH_DETECTION_MODEL  = load_default_pitch_model()

    src_info = sv.VideoInfo.from_video_path(source_video)
    frames = sv.get_video_frames_generator(source_video)

    # tiny team-classifier bootstrap (you can refine/replace)
    crops = extract_crops(source_video)
    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    template = draw_pitch(CONFIG)
    h, w = template.shape[:2]
    pitch_info = sv.VideoInfo(fps=src_info.fps, width=w, height=h, total_frames=src_info.total_frames)
    tracker = sv.ByteTrack(); tracker.reset()

    with sv.VideoSink(target_video, video_info=pitch_info) as sink:
        for frame in tqdm(frames, total=src_info.total_frames):
            res = OBJECT_DETECTION_MODEL.predict(frame, conf=0.15, verbose=False, device=device)[0]
            dets = sv.Detections.from_ultralytics(res)

            ball = dets[dets.class_id == BALL_ID]
            if ball.xyxy.size:
                ball.xyxy = sv.pad_boxes(ball.xyxy, px=10)

            others = dets[dets.class_id != BALL_ID].with_nms(0.5, class_agnostic=True)
            tracked = tracker.update_with_detections(others)

            players  = tracked[tracked.class_id == PLAYER_ID]
            keepers  = tracked[tracked.class_id == GK_ID]
            refs     = tracked[tracked.class_id == REF_ID]

            player_crops = [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]
            if player_crops:
                players.class_id = team_classifier.predict(player_crops).astype(int)
            keepers.class_id = resolve_goalkeepers_team_id(players, keepers).astype(int)

            kp_res = PITCH_DETECTION_MODEL.predict(frame, conf=0.3, verbose=False, device=device)[0]
            kp = sv.KeyPoints.from_ultralytics(kp_res)
            if kp.xy.shape[0] == 0:
                sink.write_frame(template); continue
            m = kp.confidence[0] > 0.5
            if np.sum(m) < 4:
                sink.write_frame(template); continue

            frame_ref = kp.xy[0][m]
            pitch_ref = np.array(CONFIG.vertices)[m]
            vt = ViewTransformer(source=frame_ref, target=pitch_ref)

            ball_xy    = ball.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            play_xy    = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            ref_xy     = refs.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

            pitch_ball = vt.transform_points(ball_xy)
            pitch_play = vt.transform_points(play_xy)
            pitch_ref  = vt.transform_points(ref_xy)

            canvas = template.copy()
            canvas = draw_pitch_voronoi_diagram(
                config=CONFIG,
                team_1_xy=pitch_play[players.class_id == 0],
                team_2_xy=pitch_play[players.class_id == 1],
                team_1_color=sv.Color.from_hex('00BFFF'),
                team_2_color=sv.Color.from_hex('FF1493'),
                pitch=canvas
            )
            canvas = draw_points_on_pitch(CONFIG, pitch_ball,
                                          face_color=sv.Color.WHITE, edge_color=sv.Color.BLACK,
                                          radius=10, pitch=canvas)
            canvas = draw_points_on_pitch(CONFIG, pitch_ref,
                                          face_color=sv.Color.BLACK, edge_color=sv.Color.WHITE,
                                          radius=16, pitch=canvas)
            
            canvas = draw_points_on_pitch(CONFIG, pitch_play[players.class_id == 0],
                                          face_color=sv.Color.BLUE, radius=16, pitch=canvas)
            canvas = draw_points_on_pitch(CONFIG, pitch_play[players.class_id == 1],
                                          face_color=sv.Color.RED, radius=16, pitch=canvas)
            
            sink.write_frame(canvas)
