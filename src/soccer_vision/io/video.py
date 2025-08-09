import supervision as sv

def pitch_video_info_like(template, src_info: sv.VideoInfo) -> sv.VideoInfo:
    h, w = template.shape[:2]
    return sv.VideoInfo(fps=src_info.fps, width=w, height=h, total_frames=src_info.total_frames)
