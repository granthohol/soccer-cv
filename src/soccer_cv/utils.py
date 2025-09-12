from tqdm import tqdm
import numpy as np
from .models import load_default_object_model
import supervision as sv
from typing import Optional, Sequence
from dataclasses import dataclass
import pandas as pd


def resolve_goalkeepers_team_id(players_detections: sv.Detections, goalkeepers_detections: sv.Detections):
    """
    Method to determine which goalkeeper belongs to which team. Goalkeeper is assigned to the team
    whose team centroid is closest to him. 
    """
    goalkeepers_xy = goalkeepers_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    team_0_centroid = players_xy[players_detections.class_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_detections.class_id == 1].mean(axis=0)

    goalkeepers_team_ids = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_ids.append(0 if dist_0 < dist_1 else 1)

    return np.array(goalkeepers_team_ids)


def extract_crops(source_video_path: str):
    OBJECT_DETECTION_MODEL = load_default_object_model()
    STRIDE = 30         # process 1 in every 30 frames
    PLAYER_ID = 2       # only extract crops of class_id = 2 (players)

    frame_generator = sv.get_video_frames_generator(source_video_path, stride=STRIDE)   # load every 30th frame from the video
    crops = [] 
    
    # iterate over sampled frames; tqdm just gives progress bar in terminal
    for frame in tqdm(frame_generator, desc="Collecting crops"):                    
        result = OBJECT_DETECTION_MODEL.predict(frame, conf=0.3)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        detections = detections[detections.class_id == PLAYER_ID]
        crops += [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
    return crops

@dataclass(frozen=True)
class NearestResult:
    """
    Result of nearest-player-to-ball query.

    idx  : index into players array (None if no match within radius)
    tid  : tracker id for that player (None if not provided)
    team : 0/1 if known, else None
    dist : Euclidean distance in pitch pixels (inf if no players/ball)
    """
    idx: Optional[int]
    tid: Optional[int]
    team: Optional[int]
    dist: float

    @property
    def has_match(self) -> bool:
        return self.idx is not None

def nearest_to_ball(
    ball_xy: np.ndarray,
    players_xy: np.ndarray,
    *,
    team_ids: Optional[np.ndarray] = None,
    tracker_ids: Optional[Sequence[Optional[int]]] = None,
    radius_px: float = float("inf"),
) -> NearestResult:
    """
    Find the player nearest to the (first) ball point and optionally enforce a max radius.

    Parameters
    ----------
    ball_xy : (K,2) array
        Ball coordinates in **pitch space**. If multiple, the first is used.
    players_xy : (N,2) array
        Player coordinates in **pitch space** for the same frame.
    team_ids : (N,) array, optional
        Per-player team labels (0/1 or -1/unknown). If provided, will be included.
    tracker_ids : sequence of length N, optional
        Per-player tracker ids. If provided, will be included.
    radius_px : float
        Maximum distance (pitch pixels) to accept a match. If the nearest player
        is farther than this, no match is returned.

    Returns
    -------
    NearestResult
        idx/ tid/ team/ dist; with idx=None if no match within the radius.
    """
    # Validate inputs
    if ball_xy is None or players_xy is None or ball_xy.size == 0 or players_xy.size == 0:
        return NearestResult(idx=None, tid=None, team=None, dist=float("inf"))

    # Use the first ball point
    b = np.asarray(ball_xy, dtype=np.float32).reshape(-1, 2)[0]
    P = np.asarray(players_xy, dtype=np.float32).reshape(-1, 2)

    # Squared distances for speed
    d = P - b
    d2 = np.einsum("ij,ij->i", d, d)
    j = int(np.argmin(d2))

    r2 = float(radius_px) ** 2
    if d2[j] > r2:
        # Nearest is outside radius → no possessor
        return NearestResult(idx=None, tid=None, team=None, dist=float(np.sqrt(d2[j])))

    # Fill optional fields
    tid = None
    if tracker_ids is not None and len(tracker_ids) > j and tracker_ids[j] is not None:
        tid = int(tracker_ids[j])

    team = None
    if team_ids is not None and len(team_ids) > j:
        t = int(team_ids[j])
        team = t if t in (0, 1) else None

    return NearestResult(idx=j, tid=tid, team=team, dist=float(np.sqrt(d2[j])))


def summarize_player_stats(
    csv_path: str,
    *,
    output_csv: Optional[str] = None,
    speed_hi: float = 5.0,       # m/s (≈18 km/h) “high-intensity”
    speed_sprint: float = 7.0,   # m/s (≈25 km/h) “sprint”
    accel_thr: float = 2.5       # m/s^2 (high accel magnitude)
) -> pd.DataFrame:
    """
    Read a tracking CSV (one row per player per frame) and compute per-player statistics.

    Expected columns (names case sensitive):
        frame, time_s, track_id, team_id, x_m, y_m, speed_m_s, ax_m_s2, ay_m_s2
    (Extra columns are ignored.)

    Metrics returned (per track_id):
        - team_id_mode           : most frequent team_id seen for that track
        - samples                : number of rows for that player
        - duration_s             : time span covered (max(time_s) - min(time_s))
        - total_distance_m       : path length from (x_m, y_m) diffs
        - distance_per_min_m     : total_distance_m / (duration_s / 60)
        - mean_speed_m_s         : mean of speed_m_s
        - median_speed_m_s       : median of speed_m_s
        - p95_speed_m_s          : 95th percentile of speed_m_s
        - max_speed_m_s          : max of speed_m_s
        - hi_time_s              : time with speed > speed_hi
        - sprint_time_s          : time with speed > speed_sprint
        - hi_distance_m          : ∫ v dt for v > speed_hi  (approx using per-row speed*dt)
        - accel_events           : count of rows with sqrt(ax^2+ay^2) >= accel_thr
        - max_accel_mag_m_s2     : max sqrt(ax^2+ay^2)
        - stops                  : count of transitions from moving(>0.5 m/s) → not moving

    Returns
    -------
    pd.DataFrame
        One row per player (track_id), with columns above. Also prints a pretty table.
    """
    # Load & keep only columns we need (others are ignored if present)
    cols_needed = [
        "frame", "time_s", "track_id", "team_id",
        "x_m", "y_m", "speed_m_s", "ax_m_s2", "ay_m_s2"
    ]
    df = pd.read_csv(csv_path)
    for c in cols_needed:
        if c not in df.columns:
            # Try to map can_x/can_y → x_m/y_m if meters missing
            if c == "x_m" and "x_m" not in df and "can_x" in df:
                df["x_m"] = df["can_x"].astype(float)  # fallback (unit may be pixels)
                continue
            if c == "y_m" and "y_m" not in df and "can_y" in df:
                df["y_m"] = df["can_y"].astype(float)
                continue
            if c not in df:
                raise ValueError(f"Required column '{c}' not found in CSV.")

    # Sort for sane diffs
    df = df.sort_values(["track_id", "frame", "time_s"], kind="mergesort").reset_index(drop=True)

    # Per-player summarization
    rows = []
    for tid, g in df.groupby("track_id", sort=False):
        g = g.copy()

        # Basic series (as numpy for speed)
        t = g["time_s"].to_numpy(dtype=float)
        x = g["x_m"].to_numpy(dtype=float)
        y = g["y_m"].to_numpy(dtype=float)
        v = g["speed_m_s"].to_numpy(dtype=float)
        ax = g["ax_m_s2"].to_numpy(dtype=float)
        ay = g["ay_m_s2"].to_numpy(dtype=float)

        # Time deltas aligned to *current* row (dt[0]=0)
        dt = np.diff(t, prepend=t[0])
        dt[dt < 0] = 0.0  # guard if any time glitches

        # Path length from successive positions (skip first 0 step)
        step_dx = np.diff(x, prepend=x[0])
        step_dy = np.diff(y, prepend=y[0])
        step_dist = np.hypot(step_dx, step_dy)
        # If first step is 0 (by construction), it doesn't harm sums.

        total_distance = float(np.nansum(step_dist))
        duration_s = float(max(0.0, t[-1] - t[0])) if len(t) else 0.0

        # Speed stats (robust to NaN)
        v_clean = v[~np.isnan(v)]
        mean_speed = float(np.nanmean(v)) if v_clean.size else 0.0
        median_speed = float(np.nanmedian(v)) if v_clean.size else 0.0
        p95_speed = float(np.nanpercentile(v, 95)) if v_clean.size else 0.0
        max_speed = float(np.nanmax(v)) if v_clean.size else 0.0

        # High-intensity durations (integrate by time for which v exceeds threshold)
        hi_mask = v > speed_hi
        sprint_mask = v > speed_sprint
        hi_time = float(np.nansum(dt[hi_mask])) if dt.size else 0.0
        sprint_time = float(np.nansum(dt[sprint_mask])) if dt.size else 0.0
        # Approx distance during HI using ∫ v dt
        hi_distance = float(np.nansum((v * dt)[hi_mask])) if dt.size else 0.0

        # Acceleration magnitude
        a_mag = np.hypot(ax, ay)
        accel_events = int(np.sum(a_mag >= accel_thr))
        max_accel_mag = float(np.nanmax(a_mag)) if a_mag.size else 0.0

        # Stops: moving (>0.5 m/s) → not moving transitions
        moving = v > 0.5
        stops = int(np.sum(np.logical_and(moving[:-1], ~moving[1:]))) if moving.size > 1 else 0

        # Team = modal team_id
        try:
            team_mode = int(g["team_id"].mode(dropna=True).iloc[0])
        except Exception:
            # fallback if no valid mode
            team_mode = int(g["team_id"].iloc[0]) if len(g) else -1

        rows.append(dict(
            team_id_mode=team_mode,
            track_id=int(tid),
            samples=int(len(g)),
            duration_s=round(duration_s, 2),
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

    out = pd.DataFrame(rows).sort_values(["team_id_mode", "track_id"]).reset_index(drop=True)

    # Pretty print (Markdown-style; looks nice on terminals and in READMEs)
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(out.to_string(index=False))

    if output_csv:
        out.to_csv(output_csv, index=False)

    return out

if __name__ == "__main__":
    import argparse
    import sys
    import pandas as pd

    parser = argparse.ArgumentParser(description="Summarize tracking CSV into per-player stats (Markdown table).")
    parser.add_argument("csv_path", help="Path to input CSV")
    parser.add_argument("--output-md", "-o", help="Write Markdown table to this file (default: stdout)", default=None)
    parser.add_argument("--round", type=int, default=2, help="Round numeric columns to N decimals")
    args = parser.parse_args()

    # Compute stats (expects a DataFrame return)
    df = summarize_player_stats(args.csv_path)

    # Optional rounding
    if args.round is not None:
        df = df.round(args.round)

    md = df.to_markdown(index=False)  # requires: pip install tabulate

    if args.output_md:
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write(md + "\n")
    else:
        sys.stdout.write(md + "\n")
