# RT-DETR
from ultralytics import RTDETR

data_path = "/datasets/tdt4265/Poles2025/roadpoles_v1/data.yaml"

model = RTDETR('./rtdetr/train/weights/best.pt')

model.train(
    data=data_path,
    epochs=100,
    imgsz=1280,
    batch=8,
    lr0=0.001,
    project='rtdetr_1280',
)