from roboflow import Roboflow
import os

def download():
    rf = Roboflow(api_key="viJOUNkPSJPv2zjTyGMg")
    project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
    version = project.version(12)
    dataset = version.download("yolov8")

if __name__ == "__main__":
    download()