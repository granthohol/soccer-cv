from tqdm import tqdm
import numpy as np
from .models import load_default_object_model
import supervision as sv
from typing import Optional, Sequence
from dataclasses import dataclass


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
