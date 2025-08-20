# src/soccer_cv/pipelines/voronoi2d.py
from __future__ import annotations
import numpy as np
import supervision as sv
from tqdm import tqdm
from sports.common.team import TeamClassifier
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch, draw_pitch_voronoi_diagram

from ..config import DEFAULT_CONFIG as CONFIG
from .common import (
    init_runtime, detect_ball_and_players, classify_players,
    update_homography, anchors_bottom_center, PLAYER_ID
)

def write_voronoi_2d_video(
    source_video: str,
    target_video: str,
) -> None:
    
    rt = init_runtime(source_video, want_team_classifier=True)

    frames = sv.get_video_frames_generator(source_video)

    with sv.VideoSink(target_video, video_info=rt.pitch_info) as sink:
        for i, frame in enumerate(tqdm(frames, total=rt.src_info.total_frames)):
            ball, players, refs = detect_ball_and_players(frame, rt, conf_obj=0.15)

            # classify players each frame into team 0/1
            team_ids = classify_players(frame, players, rt.team_classifier)

            # update homography (every 5 frames)
            if (rt.vt is None) or (i % 5 == 0):
                update_homography(frame, rt, keypoint_conf=0.3, min_points=4, smooth_len=5)

            # if we dont have H yet, write a blank pitch
            if rt.vt is None:
                sink.write_frame(rt.template)
                continue

            # project to pitch
            pitch_ball = rt.vt.transform_points(anchors_bottom_center(ball))
            pitch_players = rt.vt.transform_points(anchors_bottom_center(players))
            pitch_refs = rt.vt.transform_points(anchors_bottom_center(refs))

            canvas = rt.template.copy()
            canvas = draw_pitch_voronoi_diagram(
                config=CONFIG,
                team_1_xy=pitch_players[team_ids == 0],
                team_2_xy=pitch_players[team_ids == 1],
                team_1_color=sv.Color.from_hex('00BFFF'),
                team_2_color=sv.Color.from_hex('FF1493'),
                pitch=canvas
            )

            # ball, refs, players dots
            canvas = draw_points_on_pitch(CONFIG, pitch_ball, face_color=sv.Color.WHITE,
                                          edge_color=sv.Color.BLACK, radius=10, pitch=canvas)
            canvas = draw_points_on_pitch(CONFIG, pitch_refs, face_color=sv.Color.BLACK,
                                          edge_color=sv.Color.WHITE, radius=16, pitch=canvas)
            canvas = draw_points_on_pitch(CONFIG, pitch_players[team_ids == 0],
                                          face_color=sv.Color.BLUE, radius=16, pitch=canvas)
            canvas = draw_points_on_pitch(CONFIG, pitch_players[team_ids == 1],
                                          face_color=sv.Color.RED, radius=16, pitch=canvas)
            sink.write_frame(canvas)       
