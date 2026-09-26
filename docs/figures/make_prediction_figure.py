"""Compose the test images with the U-Net's saved predictions (no inference is run here).

Reads data/test/images/*.png (RGBA inputs) and data/test/labels/*.png, which hold the model's own
raw sigmoid outputs written by data.saveResult (they are predictions, not ground truth).

    python docs/figures/make_prediction_figure.py   ->   docs/figures/test_predictions.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
names = sorted(p.name for p in (ROOT / "data" / "test" / "images").glob("*.png"))

fig, axes = plt.subplots(2, len(names), figsize=(2.3 * len(names), 4.9))
for j, name in enumerate(names):
    img = Image.open(ROOT / "data" / "test" / "images" / name).convert("RGB")
    pred = np.asarray(Image.open(ROOT / "data" / "test" / "labels" / name), dtype=float) / 255.0
    axes[0, j].imshow(img)
    axes[0, j].set_title(name, fontsize=9)
    axes[1, j].imshow(pred, cmap="gray", vmin=0, vmax=1)
    for ax in axes[:, j]:
        ax.set_xticks([]); ax.set_yticks([])
axes[0, 0].set_ylabel("test image", fontsize=9)
axes[1, 0].set_ylabel("predicted mask\n(sigmoid output)", fontsize=9)
fig.tight_layout()
out = Path(__file__).with_name("test_predictions.png")
fig.savefig(out, dpi=110, facecolor="white")
# palette-quantise to keep the committed PNG small
Image.open(out).convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(out, optimize=True)
print(f"wrote {out.relative_to(ROOT)}")
