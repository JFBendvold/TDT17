import os
import random
from ultralytics import YOLO

# ----- Setup ----- 
idun = False
dataset = "roadpoles_v1"

if idun:    data_root = "/cluster/projects/vs/courses/TDT17/ad/Poles2025"
else:       data_root = "/datasets/tdt4265/Poles2025"

data_path       = os.path.join(data_root, dataset, "data.yaml")
test_set_path   = os.path.join(data_root, dataset, "test", "images")



# ----- Train Model ----- 
model = YOLO("yolo11n.pt")
results = model.train(data=data_path, epochs=100, imgsz=1280, batch=8, project="yolo", lr0=1e-3)



# ----- Save sample output ----- 
all_imgs = [os.path.join(test_set_path, f) for f in os.listdir(test_set_path)]

results = model.predict(
    source=all_imgs[5:],    # We sample the first 5 imgs
    save=True,              # saves images with boxes
    project="yolo/out",
    exist_ok=True           # don't create new numbered folder each run
)

print("Saved annotated images to: runs/export/sample_preds")