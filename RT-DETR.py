# RT-DETR
from ultralytics import RTDETR

#data_path = "/datasets/tdt4265/Poles2025/roadpoles_v1/data.yaml"
data_path = "./cluster_combined_data.yaml"

model = RTDETR('rtdetr_1280/train3/weights/best.pt')

model.train(
    data=data_path,
    epochs=10,
    imgsz=1280,
    batch=8,
    lr0=0.001,
    project='rtdetr_1280',
)