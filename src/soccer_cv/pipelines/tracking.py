# src/soccer_cv/pipelines/tracking.py
from __future__ import annotations
import os
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

from sports.annotators.soccer import (
    draw_points_on_pitch,
)

from ..config import DEFAULT_CONFIG as CONFIG
from .common import (
    init_runtime, detect_ball_and_players, classify_players,
    update_homography, anchors_bottom_center,
)

# ---------------- Tunables (mirror voronoi2d.py where sensible) ----------------
KEYPOINT_EVERY   = 5       # refresh homography every K frames
MIN_KP           = 4       # minimum keypoints to accept a homography
OBJ_CONF         = 0.15    # object detector confidence (match Voronoi)
SMOOTH_H         = 5       # homography smoothing window (frames)
TEAM0_HEX        = "00BFFF"
TEAM1_HEX        = "FF1493"

# Kalman settings (meters space)
GATE_M           = 8.0     # ignore measurements > GATE_M away from predicted pos
PROCESS_VAR      = 4.0     # process noise scale (higher = more responsive)
MEAS_VAR         = 3.0     # measurement noise (higher = smoother)

# -------------- Canonical pitch bounds (same coords as CONFIG.vertices) --------------
_VERTS = np.asarray(CONFIG.vertices, dtype=np.float32)
X_MIN, X_MAX = float(_VERTS[:, 0].min()), float(_VERTS[:, 0].max())
Y_MIN, Y_MAX = float(_VERTS[:, 1].min()), float(_VERTS[:, 1].max())

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M  = 68.0

# scale from canonical → meters
SCALE_X_M = PITCH_LENGTH_M / max(1e-6, (X_MAX - X_MIN))
SCALE_Y_M = PITCH_WIDTH_M  / max(1e-6, (Y_MAX - Y_MIN))


# ======================  Kalman for (x,y,vx,vy) in meters  ======================

@dataclass
class Kalman2D:
    """Constant-velocity Kalman filter for 2D motion in pitch meters."""
    dt: float
    process_var: float = PROCESS_VAR
    meas_var: float = MEAS_VAR

    def __post_init__(self):
        self.x = np.zeros((4, 1), dtype=np.float64)  # [x, y, vx, vy]^T
        self.P = np.eye(4, dtype=np.float64) * 10.0  # large initial uncertainty

        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1,      0],
            [0, 0, 0,      1]
        ], dtype=np.float64)

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float64)

        dt = self.dt
        q  = self.process_var
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q11 = 0.25 * dt4 * q
        q13 = 0.5  * dt3 * q
        q33 =       dt2 * q
        self.Q = np.array([
            [q11, 0,   q13, 0],
            [0,   q11, 0,   q13],
            [q13, 0,   q33, 0],
            [0,   q13, 0,   q33]
        ], dtype=np.float64)

        self.R = np.eye(2, dtype=np.float64) * self.meas_var

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: Optional[np.ndarray]):
        if z is None:
            return
        z = z.reshape(2, 1).astype(np.float64)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P

    @property
    def pos(self) -> Tuple[float, float]:
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def vel(self) -> Tuple[float, float]:
        return float(self.x[2, 0]), float(self.x[3, 0])


class TrackFilters:
    """Kalman filter per track_id with distance gating (meters) to absorb homography jitter."""
    def __init__(self, dt: float, gate_m: float = GATE_M,
                 process_var: float = PROCESS_VAR, meas_var: float = MEAS_VAR):
        self.dt = dt
        self.gate_m = gate_m
        self.process_var = process_var
        self.meas_var = meas_var
        self.filters: Dict[int, Kalman2D] = {}
        self.prev_vel: Dict[int, Tuple[float, float]] = {}

    def step(self, track_id: int, meas_xy_m: Optional[Tuple[float, float]]):
        kf = self.filters.get(track_id)
        if kf is None:
            kf = Kalman2D(self.dt, process_var=self.process_var, meas_var=self.meas_var)
            if meas_xy_m is not None:
                kf.x[0, 0] = meas_xy_m[0]
                kf.x[1, 0] = meas_xy_m[1]
            self.filters[track_id] = kf

        kf.predict()

        z = None
        if meas_xy_m is not None:
            px, py = kf.pos
            dx = meas_xy_m[0] - px
            dy = meas_xy_m[1] - py
            if math.hypot(dx, dy) <= self.gate_m:
                z = np.array([meas_xy_m[0], meas_xy_m[1]])
        kf.update(z)

        vx, vy = kf.vel
        last_v = self.prev_vel.get(track_id)
        if last_v is None:
            ax = ay = 0.0
        else:
            ax = (vx - last_v[0]) / self.dt
            ay = (vy - last_v[1]) / self.dt
        self.prev_vel[track_id] = (vx, vy)

        return (*kf.pos, *kf.vel, ax, ay)


# ---------------- Main ----------------
def write_tracking_video(source_video: str, target_video: str) -> None:
    """
    Track players, smooth positions in field meters, draw IDs + speeds on a canonical pitch, and save video.
    Also writes a metrics CSV next to the output: <target_stem>_metrics.csv

    Parameters
    ----------
    source_video : str
        Path to the input broadcast video (OpenCV-readable).
    target_video : str
        Path to the output video file. Resolution matches the library pitch template.
    """
    # Init runtime like Voronoi (provides: vt, template, pitch_info, src_info, team classifier, etc.)
    rt = init_runtime(source_video, want_team_classifier=True)
    frames = sv.get_video_frames_generator(source_video)

    # ByteTrack tracker from supervision
    tracker = sv.ByteTrack()

    # Kalman bank in meters (dt from source FPS)
    fps = max(1.0, float(rt.src_info.fps or 30.0))
    dt = 1.0 / fps
    filters = TrackFilters(dt=dt)

    # Output writer on pitch canvas (same as Voronoi)
    os.makedirs(os.path.dirname(target_video) or ".", exist_ok=True)
    with sv.VideoSink(target_video, video_info=rt.pitch_info) as sink:
        # Prepare color mapping for teams
        col_team0 = sv.Color.from_hex(TEAM0_HEX)
        col_team1 = sv.Color.from_hex(TEAM1_HEX)

        for i, frame in enumerate(tqdm(frames, total=rt.src_info.total_frames)):

            # 1) Detect & classify
            ball, players, refs = detect_ball_and_players(frame, rt, conf_obj=OBJ_CONF)
            team_ids = classify_players(frame, players, rt.team_classifier, rt.team_id_map, frame_idx=i)

            # 2) Update homography periodically
            if (rt.vt is None) or (i % KEYPOINT_EVERY == 0):
                update_homography(frame, rt, keypoint_conf=0.30, min_points=MIN_KP, smooth_len=SMOOTH_H)

            # 3) Track in pixel space (stable IDs), then anchor bottom-center points
            tracked = tracker.update_with_detections(players)  # sv.Detections with .tracker_id
            if tracked is None or tracked.xyxy.size == 0 or tracked.tracker_id is None:
                # No players → draw empty pitch and continue
                sink.write_frame(rt.template.copy())
                continue

            # 4) Team IDs for tracked detections (align lengths)
            # We classify on the original 'players', so remap by IoU to the tracked set:
            # (Supervision keeps order; when update_with_detections keeps indices, we can rely on that.
            # For robustness, compute IoU argmax mapping.)
            try:
                # Fast path: if lengths match, assume index alignment
                if team_ids is not None and len(team_ids) == tracked.xyxy.shape[0]:
                    tracked_team_ids = team_ids
                else:
                    # Robust path: match tracked boxes back to players via IoU argmax
                    tracked_team_ids = np.zeros((tracked.xyxy.shape[0],), dtype=int)
                    if players.xyxy.size:
                        ious = sv.box_iou(tracked.xyxy, players.xyxy)  # (Nt, Np)
                        nn = np.argmax(ious, axis=1)
                        tracked_team_ids = team_ids[nn] if team_ids is not None else np.zeros_like(nn)
                    else:
                        tracked_team_ids = np.zeros((tracked.xyxy.shape[0],), dtype=int)
            except Exception:
                tracked_team_ids = np.zeros((tracked.xyxy.shape[0],), dtype=int)

            # Bottom-center anchor points (pixel space), then project to canonical pitch coords
            px_pts = anchors_bottom_center(tracked)  # (N,2) in image pixels

            if rt.vt is not None and px_pts is not None and px_pts.size:
                can_pts = rt.vt.transform_points(px_pts.astype(np.float32))  # canonical pitch coords (like CONFIG.vertices)
            else:
                # No homography yet → output empty canvas
                sink.write_frame(rt.template.copy())
                continue

            # 5) Convert canonical → meters for kinematics; keep canonical for drawing
            # canonical is in template coordinate system where X ranges [X_MIN, X_MAX], Y ranges [Y_MIN, Y_MAX]
            can_x = can_pts[:, 0]
            can_y = can_pts[:, 1]
            x_m   = (can_x - X_MIN) * SCALE_X_M
            y_m   = (can_y - Y_MIN) * SCALE_Y_M

            # 6) Filter per track ID + build metrics rows
            rows = []  # collected per frame to render labels easily
            for j in range(tracked.xyxy.shape[0]):
                tid = int(tracked.tracker_id[j])
                # measurement in meters
                mx, my = float(x_m[j]), float(y_m[j])
                fx, fy, vx, vy, ax, ay = filters.step(tid, (mx, my))
                speed = math.hypot(vx, vy)
                rows.append({
                    "track_id": tid,
                    "team_id": int(tracked_team_ids[j]) if tracked_team_ids is not None else 0,
                    "can_x": float(can_x[j]),
                    "can_y": float(can_y[j]),
                    "x_m": fx, "y_m": fy,
                    "vx_m_s": vx, "vy_m_s": vy,
                    "speed_m_s": speed,
                    "ax_m_s2": ax, "ay_m_s2": ay,
                })

            # 7) Draw on pitch template
            canvas = rt.template.copy()

            # Points by team (use canonical coords)
            can_pts_arr = np.stack([can_x, can_y], axis=1).astype(np.float32)
            if np.any(tracked_team_ids == 0):
                canvas = draw_points_on_pitch(
                    CONFIG, can_pts_arr[tracked_team_ids == 0],
                    face_color=col_team0, edge_color=sv.Color.BLACK,
                    radius=14, pitch=canvas
                )
            if np.any(tracked_team_ids == 1):
                canvas = draw_points_on_pitch(
                    CONFIG, can_pts_arr[tracked_team_ids == 1],
                    face_color=col_team1, edge_color=sv.Color.BLACK,
                    radius=14, pitch=canvas
                )

            # Labels: id + speed (m/s) at the canonical location (slightly above the point)
            for r in rows:
                lx = int(round(r["can_x"]))
                ly = int(round(r["can_y"])) - 18
                label = f'id {r["track_id"]}  {r["speed_m_s"]:.1f} m/s'
                # choose color by team
                col = (255, 255, 255)
                if r["team_id"] == 0:
                    col = tuple(col_team0.as_bgr())
                elif r["team_id"] == 1:
                    col = tuple(col_team1.as_bgr())
                # shadow
                cv2.putText(canvas, label, (lx+1, ly+1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
                # text
                cv2.putText(canvas, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

            # 8) Write frame
            sink.write_frame(canvas)

            # 9) Append to metrics file (streaming write to avoid big RAM)
            #    We create the CSV on first write. Using Python CSV to avoid pandas dependency.
            metrics_path = os.path.splitext(target_video)[0] + "_metrics.csv"
            header = [
                "frame", "time_s", "track_id", "team_id",
                "can_x", "can_y",
                "x_m", "y_m", "vx_m_s", "vy_m_s", "speed_m_s", "ax_m_s2", "ay_m_s2"
            ]
            exists = os.path.exists(metrics_path)
            os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
            import csv
            with open(metrics_path, "a", newline="") as f:
                w = csv.writer(f)
                if not exists:
                    w.writerow(header)
                for r in rows:
                    w.writerow([
                        i, i / fps, r["track_id"], r["team_id"],
                        r["can_x"], r["can_y"],
                        r["x_m"], r["y_m"], r["vx_m_s"], r["vy_m_s"], r["speed_m_s"], r["ax_m_s2"], r["ay_m_s2"]
                    ])
