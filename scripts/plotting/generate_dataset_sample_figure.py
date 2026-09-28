from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base_dir = Path(__file__).resolve().parents[2]
img_dir = base_dir / "paper" / "images"
out = img_dir / "dataset_samples.png"
out.parent.mkdir(parents=True, exist_ok=True)

cityscapes = img_dir / "PHOTO-2026-09-27-22-12-16.jpg"
synscapes = img_dir / "PHOTO-2026-09-27-22-12-16 3.jpg"
gta = img_dir / "PHOTO-2026-09-27-22-12-16 2.jpg"

images = [
    ("Cityscapes", cityscapes),
    ("Synscapes", synscapes),
    ("GTA-V", gta),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=220)
fig.patch.set_facecolor("#f8f8f8")

for ax, (label, path) in zip(axes, images):
    img = plt.imread(path)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(label, fontsize=14, fontweight="bold", pad=10)

fig.tight_layout()
fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
fig.savefig(str(out.with_suffix(".pdf")), bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved {out}")
print(f"Saved {out.with_suffix('.pdf')}")
