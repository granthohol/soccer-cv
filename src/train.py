#!/usr/bin/env python3
import gc
import torch
from ultralytics import YOLO

def clear_gpu():
    """Free up as much GPU RAM as possible before training."""
    gc.collect()                      
    torch.cuda.empty_cache()          

def train():
    clear_gpu()

    # 1) Load a pre-trained YOLOv8x model
    model = YOLO("yolov8x.pt")        

    # 2) Kick off training with your exact specs
    model.train(
        data="data/data.yaml",   # your data config
        epochs=50,
        batch=6,
        imgsz=1280,
        half=True,               # enable FP16
        device=0                 # choose GPU 0
    )

if __name__ == "__main__":
    train()
