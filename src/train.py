
from ultralytics import YOLO
import gc
import torch

gc.collect()                      
torch.cuda.empty_cache() 
detection_model = YOLO("yolov8s.pt")        


detection_model.train(
    data="football-players-detection-12/data.yaml",
    epochs=100,             
    batch=3,                
    imgsz=1280,            
    half=True,            
    augment=True,          
    lr0=0.001,              
    weight_decay=0.0005,    
    patience=20,            
    optimizer="Adam",       
    device=0                
)