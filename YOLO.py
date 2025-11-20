import os
from ultralytics import YOLO


# ----- Setup ----- 
idun = False
dataset = "roadpoles_v1"

if idun:    data_root = "/cluster/projects/vs/courses/TDT17/ad/Poles2025"
else:       data_root = "/datasets/tdt4265/Poles2025"

data_path       = os.path.join(data_root, dataset, "data.yaml")
test_set_path   = os.path.join(data_root, dataset, "test", "images")
output_dir      = "yolo/nano"



# ----- Train Model ----- 
model = YOLO("yolo11n.pt")
results = model.train(data=data_path, epochs=200, imgsz=1280, batch=8, project=output_dir, lr0=1e-3)

# We use the best performing model for analysis
best_model_path= os.path.join(output_dir, "train", "weights", "best.pt")
best_model = YOLO(best_model_path)



# ----- Save sample output ----- 
all_imgs = [os.path.join(test_set_path, f) for f in os.listdir(test_set_path)]

results = best_model.predict(
    source=all_imgs,
    save=True,
    project=os.path.join(output_dir, "out"),
    exist_ok=True
)

print("Saved annotated images to: runs/export/sample_preds")



# ----- Evaluation on Testset -----
metrics = best_model.val(
    data=data_path,
    split="val", 
    imgsz=1280,
    batch=8
)

metrics_path = os.path.join(output_dir, "out", "test_metrics.txt")

with open(metrics_path, "w") as f:
    f.write("--- Test Set Performance ---\n")
    f.write(f"Precision:       {metrics.box.mp:.6f}\n")
    f.write(f"Recall:          {metrics.box.mr:.6f}\n")
    f.write(f"mAP@50:          {metrics.box.map50:.6f}\n")
    f.write(f"mAP@0.5:0.95:    {metrics.box.map:.6f}\n")

print(f"Saved metrics to: {metrics_path}")
