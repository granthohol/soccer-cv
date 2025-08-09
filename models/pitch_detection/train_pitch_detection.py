from ultralytics import YOLO
import gc
import torch

torch.cuda.empty_cache() 
gc.collect()                      
model = YOLO("yolov8x-pose.pt")

model.train(
    data="/kaggle/working/football-field-detection-12/data.yaml",
    batch=16,
    epochs=100,
    imgsz=640,
    mosaic=0.0,
    plots=True,
    amp=True,
    optimizer='AdamW',
    max_det=40,
    device='0',
    
    project="soccer-cv-pitch=keypoints"
)