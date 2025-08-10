from tqdm import tqdm
import numpy as np
from models import load_default_object_model
import supervision as sv


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

