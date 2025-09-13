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
                    
                    
def summarize_player_stats(
    csv_path: str,
    *,
    output_csv: Optional[str] = None,
    speed_hi: float = 5.0,       # m/s (≈18 km/h) “high-intensity”
    speed_sprint: float = 7.0,   # m/s (≈25 km/h) “sprint”
    accel_thr: float = 2.5       # m/s^2 (high accel magnitude threshold)
) -> pd.DataFrame:
    """
    Read a tracking CSV (one row per player per frame, created using 'write_tracking_video())
    and compute per-player statistics.

    Expected columns (names case sensitive):
        frame, time_s, track_id, team_id, x_m, y_m, speed_m_s, ax_m_s2, ay_m_s2
    (If x_m/y_m are missing, will fall back to can_x/can_y.)

    Output columns (per track_id):
        - team_id_mode
        - track_id
        - total_distance_m
        - distance_per_min_m
        - mean_speed_m_s
        - median_speed_m_s
        - p95_speed_m_s
        - max_speed_m_s
        - hi_time_s
        - sprint_time_s
        - hi_distance_m
        - accel_events           
        - max_accel_mag_m_s2
        - stops
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pandas is required for summarize_player_stats(). "
            "Install with `pip install soccer-cv[data]` or `pip install pandas`."
        ) from e
    
    # ---------- Load & normalize required columns ----------
    cols_needed = [
        "frame", "time_s", "track_id", "team_id",
        "x_m", "y_m", "speed_m_s", "ax_m_s2", "ay_m_s2"
    ]
    df = pd.read_csv(csv_path)

    for c in cols_needed:
        if c not in df.columns:
            # Fallback for positions
            if c == "x_m" and "can_x" in df.columns:
                df["x_m"] = df["can_x"].astype(float)
                continue
            if c == "y_m" and "can_y" in df.columns:
                df["y_m"] = df["can_y"].astype(float)
                continue
            if c not in df.columns:
                raise ValueError(f"Required column '{c}' not found in CSV.")

    # Sort for stable diffs
    df = df.sort_values(["track_id", "frame", "time_s"], kind="mergesort").reset_index(drop=True)

    # ---------- Select top-10 track_ids per team by presence (row count) ----------
    def _mode_safe(s: pd.Series) -> int:
        try:
            m = s.mode(dropna=True)
            if len(m):
                return int(m.iloc[0])
            s2 = s.dropna()
            return int(s2.iloc[0]) if len(s2) else -1
        except Exception:
            return -1

    pre = (
        df.groupby("track_id", sort=False)
          .agg(samples=("track_id", "size"),
               team_id_mode=("team_id", _mode_safe))
          .reset_index()
    )

    top_ids = []
    for tm in (0, 1):
        ids = (pre.loc[pre.team_id_mode == tm]
                  .sort_values("samples", ascending=False)
                  .head(10)["track_id"]
                  .tolist())
        top_ids.extend(ids)

    df = df[df["track_id"].isin(top_ids)].copy()

    # ---------- Per-player summarization ----------
    rows = []

    # Helper: stricter acceleration event counter
    # Counts a burst only if:
    #   - a_mag >= accel_thr AND speed > 1.5 m/s (gate)
    #   - sustained for >= 0.12 s
    #   - speed gain across the burst >= 0.8 m/s
    #   - 0.30 s cooldown after each accepted burst
    def _count_accel_bursts(t: np.ndarray, v: np.ndarray, a_mag: np.ndarray) -> int:
        if t.size < 2:
            return 0
        n = len(t)
        i = 0
        events = 0
        MIN_DUR = 0.12       # seconds
        MIN_DV  = 0.8        # m/s
        SPEED_GATE = 1.5     # m/s
        COOLDOWN = 0.30      # seconds
        while i < n:
            if a_mag[i] >= accel_thr and v[i] > SPEED_GATE:
                # grow segment
                j = i + 1
                while j < n and a_mag[j] >= accel_thr and v[j] > SPEED_GATE:
                    j += 1
                t0, t1 = t[i], t[j-1]
                dur = max(0.0, float(t1 - t0))
                dv  = float(v[j-1] - v[i])
                if dur >= MIN_DUR and dv >= MIN_DV:
                    events += 1
                    # cooldown: skip ahead until time passes COOLDOWN beyond t1
                    while j < n and t[j] < t1 + COOLDOWN:
                        j += 1
                i = j
            else:
                i += 1
        return events

    for tid, g in df.groupby("track_id", sort=False):
        g = g.copy()

        # Basic series
        t  = g["time_s"].to_numpy(dtype=float)
        x  = g["x_m"].to_numpy(dtype=float)
        y  = g["y_m"].to_numpy(dtype=float)
        v  = g["speed_m_s"].to_numpy(dtype=float)
        ax = g["ax_m_s2"].to_numpy(dtype=float)
        ay = g["ay_m_s2"].to_numpy(dtype=float)

        # Time deltas aligned to current rows (dt[0] = 0)
        dt = np.diff(t, prepend=t[0])
        dt[dt < 0] = 0.0  # guard time glitches

        # Path length
        step_dx = np.diff(x, prepend=x[0])
        step_dy = np.diff(y, prepend=y[0])
        step_dist = np.hypot(step_dx, step_dy)
        total_distance = float(np.nansum(step_dist))

        # Duration for internal rate calcs (not returned)
        duration_s = float(max(0.0, t[-1] - t[0])) if len(t) else 0.0

        # Speed stats
        v_clean = v[~np.isnan(v)]
        mean_speed   = float(np.nanmean(v)) if v_clean.size else 0.0
        median_speed = float(np.nanmedian(v)) if v_clean.size else 0.0
        p95_speed    = float(np.nanpercentile(v, 95)) if v_clean.size else 0.0
        max_speed    = float(np.nanmax(v)) if v_clean.size else 0.0

        # High-intensity time & distance
        hi_mask     = v > speed_hi
        sprint_mask = v > speed_sprint
        hi_time     = float(np.nansum(dt[hi_mask])) if dt.size else 0.0
        sprint_time = float(np.nansum(dt[sprint_mask])) if dt.size else 0.0
        hi_distance = float(np.nansum((v * dt)[hi_mask])) if dt.size else 0.0

        # Acceleration
        a_mag = np.hypot(ax, ay)
        accel_events = _count_accel_bursts(t, v, a_mag)
        max_accel_mag = float(np.nanmax(a_mag)) if a_mag.size else 0.0

        # Stops: moving (>0.5 m/s) → not moving
        moving = v > 0.5
        stops = int(np.sum(np.logical_and(moving[:-1], ~moving[1:]))) if moving.size > 1 else 0

        # Team = modal team_id
        try:
            team_mode = int(g["team_id"].mode(dropna=True).iloc[0])
        except Exception:
            team_mode = int(g["team_id"].iloc[0]) if len(g) else -1

        rows.append(dict(
            team_id_mode=team_mode,
            track_id=int(tid),
            total_distance_m=round(total_distance, 2),
            distance_per_min_m=round(total_distance / (duration_s / 60.0), 2) if duration_s > 0 else 0.0,
            mean_speed_m_s=round(mean_speed, 2),
            median_speed_m_s=round(median_speed, 2),
            p95_speed_m_s=round(p95_speed, 2),
            max_speed_m_s=round(max_speed, 2),
            hi_time_s=round(hi_time, 2),
            sprint_time_s=round(sprint_time, 2),
            hi_distance_m=round(hi_distance, 2),
            accel_events=int(accel_events),
            max_accel_mag_m_s2=round(max_accel_mag, 2),
            stops=int(stops),
        ))

    out = pd.DataFrame(rows)

    # Keep only top-10 per team *again* at the summary level (in case some track_ids flipped team_mode)
    # We determine "top" by presence in the original df (row counts).
    presence = df.groupby("track_id").size().rename("samples")
    out = out.merge(presence, on="track_id", how="left")
    filtered = []
    for tm in (0, 1):
        sub = out[out["team_id_mode"] == tm].sort_values("samples", ascending=False).head(10)
        filtered.append(sub)
    out = pd.concat(filtered, axis=0).sort_values(["team_id_mode", "track_id"]).reset_index(drop=True)
    out = out.drop(columns=["samples"], errors="ignore")  # drop helper column

    # Pretty print
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(out.to_string(index=False))

    if output_csv:
        out.to_csv(output_csv, index=False)

    return out
