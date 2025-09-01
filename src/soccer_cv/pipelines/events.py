# src/soccer_cv/pipelines/events.py
from __future__ import annotations
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Deque
from collections import deque

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

from ..config import DEFAULT_CONFIG as CONFIG
from .common import (
    init_runtime,
    detect_ball_and_players,
    classify_players,
    update_homography,
    anchors_bottom_center,
)
from ..utils import nearest_to_ball  # unified helper (returns tid/team/distance)

# ---------------- Tunables (defaults; override via function kwargs) ----------------
OBJ_CONF              = 0.15
KEYPOINT_EVERY        = 5
SMOOTH_H              = 5
MIN_KP                = 4

HOLD_RADIUS_PX        = 60.0   # assign ball to nearest player within this pitch distance
LOST_GRACE_FRAMES     = 12     # keep last holder this many frames if ball missing

PASS_GAP_FRAMES       = 30     # max frames between holder switches to count as a pass
MIN_PASS_MOVE_PX      = 20.0   # min ball travel between holders (pitch px)

# --- Shot gating (new/updated) ---
SHOT_SPEED_PX_S       = 650.0   # raise baseline speed
SHOT_ANGLE_DEG        = 22.0    # narrower aim cone
SHOT_SUSTAIN_FRAMES   = 4       # speed must hold for S recent frames
SHOT_COOLDOWN_FRAMES  = 20      # don't re-fire too soon after a shot
SHOT_FINAL_THIRD_FRAC = 0.33    # must be inside attacking third toward goal
SHOT_RELEASE_WITHIN   = 3       # or within N frames of last holder change
SHOT_RELEASE_SEP_MULT = 1.3     # or separated from ALL players by 1.3 * hold_radius
SHOT_MIN_RADIAL_PX_S  = 250.0   # velocity component toward goal (px/s)
VEL_SMOOTH            = 4       # compute velocity over the last N samples

OUT_MARGIN_PX         = 0.0    # allow a small tolerance outside bounds

# Canonical pitch bounds from config
_VERTS = np.asarray(CONFIG.vertices, dtype=np.float32)
X_MIN, X_MAX = float(_VERTS[:, 0].min()), float(_VERTS[:, 0].max())
Y_MIN, Y_MAX = float(_VERTS[:, 1].min()), float(_VERTS[:, 1].max())
X_MID = 0.5 * (X_MIN + X_MAX)
Y_MID = 0.5 * (Y_MIN + Y_MAX)

LEFT_GOAL  = np.array([X_MIN, Y_MID], dtype=np.float32)
RIGHT_GOAL = np.array([X_MAX, Y_MID], dtype=np.float32)


# ---------------- Small helpers ----------------
@dataclass
class Holder:
    tid: Optional[int]   # tracker id
    team: Optional[int]  # 0/1 or None/-1
    frame: int           # when this holder became active


def _nearest_player(ball_xy: np.ndarray,
                    players_xy: np.ndarray,
                    tids: np.ndarray,
                    team_ids: np.ndarray,
                    radius_px: float) -> tuple[Optional[int], Optional[int], float]:
    """
    Thin wrapper around utils.nearest_to_ball for this pipeline’s shape.
    Returns (tid, team, distance_px)
    """
    res = nearest_to_ball(
        ball_xy,
        players_xy,
        tracker_ids=tids,
        team_ids=team_ids,
        radius_px=radius_px,
    )
    return res.tid, (res.team if res.team is not None else None), (float(res.dist) if res.dist is not None else float("inf"))


def _vec_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees between 2 vectors (handles zeros)."""
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 180.0
    cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _inside_pitch(xy: np.ndarray, margin: float = 0.0) -> bool:
    if xy is None or xy.size != 2:
        return False
    x, y = float(xy[0]), float(xy[1])
    return (X_MIN - margin) <= x <= (X_MAX + margin) and (Y_MIN - margin) <= y <= (Y_MAX + margin)


def _write_csv(path: str, rows: List[Dict]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        # still create file with header
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["frame", "time_s", "event", "team", "player_tid", "x", "y", "speed_px_s", "extra"])
            w.writeheader()
        return path
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def _write_jsonl(path: str, rows: List[Dict]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


# ---------------- Main API ----------------
def detect_events_to_file(
    source_video: str,
    out_path: str,
    *,
    # tuning knobs (override per call as needed)
    obj_conf: float = OBJ_CONF,
    keypoint_every: int = KEYPOINT_EVERY,
    min_kp: int = MIN_KP,
    smooth_h: int = SMOOTH_H,
    hold_radius_px: float = HOLD_RADIUS_PX,
    lost_grace_frames: int = LOST_GRACE_FRAMES,
    pass_gap_frames: int = PASS_GAP_FRAMES,
    min_pass_move_px: float = MIN_PASS_MOVE_PX,
    shot_speed_px_s: float = SHOT_SPEED_PX_S,
    shot_angle_deg: float = SHOT_ANGLE_DEG,
    vel_smooth: int = VEL_SMOOTH,
    out_margin_px: float = OUT_MARGIN_PX,
    fmt: str = "csv",   # "csv" or "jsonl"
) -> str:
    """
    Detect **soccer events** from a broadcast clip and write them to disk.

    Events
    ------
    - `pass`      : holder switches to a teammate within a short time window and ball moved enough
    - `turnover`  : holder team changes
    - `shot`      : ball moving fast toward either goal (speed + angle)
    - `ball_out`  : ball leaves the pitch bounds

    Output schema (CSV/JSONL)
    -------------------------
    frame,time_s,event,team,player_tid,x,y,speed_px_s,extra_json

    Returns
    -------
    str
        The path written (`out_path`).
    """
    rt = init_runtime(source_video, want_team_classifier=True)
    if not hasattr(rt, "team_id_map"):
        rt.team_id_map = {}

    frames = sv.get_video_frames_generator(source_video)
    fps = float(rt.pitch_info.fps if rt.pitch_info.fps else 30.0)

    # state
    rows: List[Dict] = []
    holder = Holder(tid=None, team=None, frame=-10**9)
    last_ball_xy: Optional[np.ndarray] = None
    last_holder_ball_xy: Optional[np.ndarray] = None
    last_holder_change_frame: int = -10**9
    frames_since_seen_ball = 0

    # for velocity (px/s) we keep recent coordinates
    ball_hist: Deque[Tuple[int, np.ndarray]] = deque(maxlen=max(2, vel_smooth))
    last_shot_frame: int = -10**9
    speed_hist: Deque[float] = deque(maxlen=max(2, SHOT_SUSTAIN_FRAMES))


    def add_event(event: str, frame_idx: int, team: Optional[int], tid: Optional[int],
                  xy: Optional[np.ndarray], speed_px_s: float, extra: Dict) -> None:
        t = frame_idx / fps
        x, y = (float(xy[0]), float(xy[1])) if xy is not None and xy.size == 2 else ("", "")
        rows.append({
            "frame": frame_idx,
            "time_s": round(t, 3),
            "event": event,
            "team": (int(team) if team in (0, 1) else -1),
            "player_tid": (int(tid) if tid is not None else ""),
            "x": (round(x, 2) if x != "" else ""),
            "y": (round(y, 2) if y != "" else ""),
            "speed_px_s": round(float(speed_px_s), 2) if np.isfinite(speed_px_s) else 0.0,
            "extra": json.dumps(extra or {}, separators=(",", ":")),
        })

    for i, frame in enumerate(tqdm(frames, total=rt.src_info.total_frames)):
        # 1) detect / track / classify
        ball, players, _ = detect_ball_and_players(frame, rt, conf_obj=obj_conf)
        team_ids = classify_players(frame, players, rt.team_classifier, rt.team_id_map, frame_idx=i)

        # 2) homography refresh
        if (rt.vt is None) or (i % keypoint_every == 0):
            update_homography(frame, rt, keypoint_conf=0.30, min_points=min_kp, smooth_len=smooth_h)

        # 3) project to pitch
        pitch_ball = np.empty((0, 2), np.float32)
        pitch_players = np.empty((0, 2), np.float32)
        if rt.vt is not None:
            if ball.xyxy.size:
                bb = anchors_bottom_center(ball)
                if bb.size:
                    pitch_ball = rt.vt.transform_points(bb)
            if players.xyxy.size:
                pitch_players = rt.vt.transform_points(anchors_bottom_center(players))

        # extract tids robustly
        tids = getattr(players, "tracker_id", None)
        if tids is None:
            tids = np.array([None] * len(players), dtype=object)
        else:
            tids = np.asarray(tids, dtype=object)
            if tids.shape[0] != len(players):
                tids = np.resize(tids, (len(players),))

        # 4) holder inference
        new_tid: Optional[int] = None
        new_team: Optional[int] = None
        dist = float("inf")

        has_ball = pitch_ball.size != 0
        if has_ball:
            frames_since_seen_ball = 0
            new_tid, new_team, dist = _nearest_player(
                pitch_ball, pitch_players, tids, team_ids, hold_radius_px
            )
        else:
            frames_since_seen_ball += 1
            # grace period: keep last holder for a short time if ball missing
            if frames_since_seen_ball <= lost_grace_frames:
                new_tid, new_team = holder.tid, holder.team

        # 5) ball velocity
        ball_xy = pitch_ball.reshape(-1, 2)[0] if pitch_ball.size else None
        if ball_xy is not None:
            ball_hist.append((i, ball_xy.copy()))

        speed_px_s = 0.0
        if len(ball_hist) >= 2:
            (f0, p0), (f1, p1) = ball_hist[0], ball_hist[-1]
            dt = max(1, f1 - f0) / fps
            speed_px_s = float(np.linalg.norm(p1 - p0) / dt) if dt > 0 else 0.0
        v_vec = np.zeros(2, dtype=np.float32)
        if len(ball_hist) >= 2:
            v_vec = (ball_hist[-1][1] - ball_hist[-2][1])  # last-step direction (px/frame)

        # 6) event logic

        # (a) holder change → pass or turnover
        if new_tid is not None and (new_tid != holder.tid):
            same_team = (holder.team in (0, 1)) and (new_team in (0, 1)) and (holder.team == new_team)

            # distance traveled since last holder change
            moved_ok = True
            if last_holder_ball_xy is not None and ball_xy is not None:
                moved_ok = (np.linalg.norm(ball_xy - last_holder_ball_xy) >= min_pass_move_px)

            # how long since previous holder
            gap_ok = (i - last_holder_change_frame) <= pass_gap_frames

            if same_team and moved_ok and gap_ok:
                add_event(
                    "pass", i, new_team, new_tid, ball_xy, speed_px_s,
                    extra={"from": int(holder.tid) if holder.tid is not None else None,
                           "to": int(new_tid)}
                )
            elif (holder.team in (0, 1)) and (new_team in (0, 1)) and (holder.team != new_team):
                add_event("turnover", i, new_team, new_tid, ball_xy, speed_px_s, extra={})

            # update holder state
            holder = Holder(tid=new_tid, team=new_team, frame=i)
            last_holder_change_frame = i
            last_holder_ball_xy = (ball_xy.copy() if ball_xy is not None else None)

        # seed holder if none yet
        if holder.tid is None and new_tid is not None:
            holder = Holder(tid=new_tid, team=new_team, frame=i)
            last_holder_change_frame = i
            last_holder_ball_xy = (ball_xy.copy() if ball_xy is not None else None)

        # (b) shot: sustained fast ball, aimed at a goal, in attacking third, with release cue
        if ball_xy is not None and holder.team in (0, 1):

            # average velocity vector over the smoothing window
            v_avg = np.zeros(2, dtype=np.float32)
            speed_avg = 0.0
            if len(ball_hist) >= 2:
                (f0, p0), (f1, p1) = ball_hist[0], ball_hist[-1]
                dt = max(1, f1 - f0) / fps
                if dt > 0:
                    v_avg = (p1 - p0) / dt           # px / s
                    speed_avg = float(np.linalg.norm(v_avg))

            # keep recent instantaneous speeds for "sustain"
            if len(ball_hist) >= 2:
                inst_dt = max(1, ball_hist[-1][0] - ball_hist[-2][0]) / fps
                if inst_dt > 0:
                    inst_v = (ball_hist[-1][1] - ball_hist[-2][1]) / inst_dt
                    speed_hist.append(float(np.linalg.norm(inst_v)))

            # choose nearer goal for angle test
            to_left  = LEFT_GOAL  - ball_xy
            to_right = RIGHT_GOAL - ball_xy
            ang_l = _vec_angle_deg(v_avg, to_left)
            ang_r = _vec_angle_deg(v_avg, to_right)
            towards = "left" if ang_l < ang_r else "right"
            angle_deg = min(ang_l, ang_r)

            # location: ball must be in attacking third toward the chosen goal
            third_w = (X_MAX - X_MIN) * SHOT_FINAL_THIRD_FRAC
            if towards == "left":
                location_ok = (ball_xy[0] <= X_MIN + third_w)
                goal_vec = to_left
            else:
                location_ok = (ball_xy[0] >= X_MAX - third_w)
                goal_vec = to_right

            # radial component toward goal
            radial_speed = 0.0
            nv = np.linalg.norm(goal_vec)
            if nv > 1e-6:
                radial_speed = float(np.dot(v_avg, goal_vec / nv))  # px/s toward goal (signed)

            # release cue: (a) very near after holder change, or (b) ball separated from everyone
            near_after_kick = (i - last_holder_change_frame) <= SHOT_RELEASE_WITHIN
            sep_ok = False
            if pitch_players.size:
                dists = np.linalg.norm(pitch_players - ball_xy[None, :], axis=1)
                sep_ok = bool(dists.size and float(dists.min()) >= SHOT_RELEASE_SEP_MULT * hold_radius_px)
            release_ok = near_after_kick or sep_ok

            # sustain gate on recent speeds
            sustain_ok = (len(speed_hist) >= SHOT_SUSTAIN_FRAMES and min(speed_hist) >= SHOT_SPEED_PX_S * 0.95)

            # final gates
            speed_ok   = (speed_avg >= SHOT_SPEED_PX_S)
            angle_ok   = (angle_deg <= SHOT_ANGLE_DEG)
            radial_ok  = (radial_speed >= SHOT_MIN_RADIAL_PX_S)
            cooldown_ok = (i - last_shot_frame >= SHOT_COOLDOWN_FRAMES)

            if all([speed_ok, angle_ok, sustain_ok, location_ok, radial_ok, release_ok, cooldown_ok]):
                add_event("shot", i, holder.team, holder.tid, ball_xy, speed_avg, extra={"towards": towards})
                last_shot_frame = i


        # (c) ball out
        if ball_xy is not None and not _inside_pitch(ball_xy, margin=out_margin_px):
            add_event("ball_out", i, holder.team, holder.tid, ball_xy, speed_px_s, extra={})

        # remember last ball
        last_ball_xy = ball_xy if ball_xy is not None else last_ball_xy

        # 7) homography refresh cadence
        if (rt.vt is None) or (i % keypoint_every == 0):
            # already done above; kept for symmetry
            pass

    # 8) write
    ext = (fmt or "").lower()
    if ext == "jsonl" or out_path.lower().endswith(".jsonl"):
        return _write_jsonl(out_path, rows)
    else:
        return _write_csv(out_path, rows)
