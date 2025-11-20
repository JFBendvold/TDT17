import os
import time
from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------
dataset_root = "/datasets/tdt4265/Poles2025/roadpoles_v1"
test_images_path = os.path.join(dataset_root, "test", "images")

model_small_path  = "yolo/small/train/weights/best.pt"
model_nano_path   = "yolo/nano/train/weights/best.pt"
model_medium_path = "yolo/medium/train/weights/best.pt"

test_models = {
    "small": model_small_path,
    "nano":  model_nano_path,
    "medium": model_medium_path,
}

runtime = 60 

# -----------------------------

# Load test images
test_images = [
    os.path.join(test_images_path, f)
    for f in os.listdir(test_images_path)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if len(test_images) == 0:
    raise ValueError("No test images found!")

print(f"Loaded {len(test_images)} test images.")


def benchmark_model(model_name, model_path):
    print(f"\n--- Benchmarking {model_name} ---")
    model = YOLO(model_path)

    start_time = time.time()
    end_time   = start_time + runtime

    count = 0
    idx = 0

    while time.time() < end_time:
        # speed through images until done
        img_path = test_images[idx]
        model.predict(
            source=img_path,
            imgsz=1280,
            verbose=False,   # disable output for MAXIMUM SPEED
            device=0
        )

        count += 1
        idx = (idx + 1) % len(test_images)

    total_time = time.time() - start_time
    sec_per_image = total_time / count if count > 0 else float("inf")

    print(f"Processed {count} images in {total_time:.2f}s")
    print(f"{model_name} speed: {sec_per_image:.6f} s/image")
    print(f"{model_name} max fps: {1/sec_per_image:.6f} image/s")

    return sec_per_image, count, total_time


# -----------------------------
# RUN BENCHMARKS
# -----------------------------
results = {}

for name, path in test_models.items():
    results[name] = benchmark_model(name, path)

print("\n\n=== FINAL RESULTS ===")
for name, (sec_per_img, count, total_t) in results.items():
    print(f"{name}: {sec_per_img:.6f} s/image ({count} images in {total_t:.2f}s)")
