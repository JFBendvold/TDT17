import matplotlib.pyplot as plt
import numpy as np

# --- Data ---
models = ["small", "nano"]
metrics = ["Precision", "Recall", "mAP@50", "mAP@0.5:0.95"]

values_small = [0.969003, 0.829958, 0.967884, 0.718106]
values_nano  = [0.871145, 0.814159, 0.927530, 0.603647]

# --- Setup ---
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(11, 6))
plt.style.use("ggplot")  # cleaner style

# Colors
small_color = "#4C72B0"   # blue
nano_color  = "#DD8452"   # orange

# --- Bars ---
bars_small = plt.bar(x - width/2, values_small, width, label="small", color=small_color)
bars_nano  = plt.bar(x + width/2, values_nano,  width, label="nano",  color=nano_color)

# --- Labels on top of bars ---
for bar in bars_small + bars_nano:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.015,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

# --- Axes and titles ---
plt.xticks(x, metrics, fontsize=11)
plt.ylabel("Score", fontsize=12)
plt.title("Model Performance Comparison (small vs nano)", fontsize=14, weight="bold")

plt.ylim(0, 1.1)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.legend(fontsize=11)
plt.tight_layout()

# --- Show plot ---
plt.show()
