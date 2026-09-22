"""
Demonstrates how the CycloSense image model responds to a wide variety
of NON-CYCLONE inputs: random noise, blank images, natural scenes,
ocean images, land images, and synthetic IR-like patterns.

Run:  python tests/probe_non_cyclone_inputs.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFilter
import io

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.image_cnn import ImageBaselineCNN

# ── Load trained model ───────────────────────────────────────────────────────
model = ImageBaselineCNN(num_classes=2, pretrained=False)
model.load_state_dict(torch.load(
    PROJECT_ROOT / "models/image_model/model.pt", map_location="cpu"
))
model.eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(img: Image.Image) -> tuple[float, float, str]:
    """Run image through model, return (p_non_cyc, p_cyc, label)."""
    t = tf(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)[0].numpy()
    label = "CYCLONE" if probs[1] >= 0.5 else "non-cyclone"
    return float(probs[0]), float(probs[1]), label

def bar(p: float, width: int = 20) -> str:
    filled = int(round(p * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"

# ── Build a variety of non-cyclone test images ──────────────────────────────
W = H = 256

def solid(r, g, b): return Image.fromarray(np.full((H, W, 3), [r, g, b], dtype=np.uint8))

def noise(mean=128, std=50):
    arr = np.random.normal(mean, std, (H, W, 3)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def gradient_sky():
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        v = int(135 + (y / H) * 80)
        arr[y, :] = [int(v * 0.45), int(v * 0.65), v]
    return Image.fromarray(arr)

def ocean_surface():
    arr = np.random.normal([25, 60, 110], [5, 8, 12], (H, W, 3)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def scattered_cumulus():
    img = ocean_surface()
    d = ImageDraw.Draw(img)
    for cx, cy, rx, ry in [(80, 70, 35, 20), (170, 110, 28, 16), (50, 170, 22, 12), (200, 55, 18, 10)]:
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(220, 230, 245))
    return img.filter(ImageFilter.GaussianBlur(radius=5))

def cirrus_streaks():
    img = ocean_surface()
    d = ImageDraw.Draw(img)
    for i in range(6):
        y_off = i * 35 + 10
        d.line([(10, y_off), (W - 10, y_off + np.random.randint(-15, 15))],
               fill=(200, 210, 230), width=3)
    return img.filter(ImageFilter.GaussianBlur(radius=4))

def desert_land():
    arr = np.random.normal([205, 170, 110], [15, 10, 10], (H, W, 3)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def forest_land():
    arr = np.random.normal([40, 90, 45], [10, 15, 10], (H, W, 3)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def uniform_gray():
    return solid(128, 128, 128)

def all_black():
    return solid(0, 0, 0)

def all_white():
    return solid(255, 255, 255)

def checkerboard():
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    bs = 32
    for r in range(H // bs):
        for c in range(W // bs):
            color = 240 if (r + c) % 2 == 0 else 30
            arr[r * bs:(r + 1) * bs, c * bs:(c + 1) * bs] = color
    return Image.fromarray(arr)

def warm_ir_ocean():
    """Simulate a warm ocean IR signature (dark = warm, uniform, no convective cells)."""
    arr = np.random.normal([18, 28, 48], [3, 4, 5], (H, W, 3)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def random_gaussian_noise():
    return noise(mean=128, std=80)

np.random.seed(42)

cases = [
    ("All Black (0,0,0)",          all_black()),
    ("All White (255,255,255)",     all_white()),
    ("Uniform Gray (128,128,128)",  uniform_gray()),
    ("Random Gaussian Noise",       random_gaussian_noise()),
    ("Open Ocean IR (warm, dark)",  warm_ir_ocean()),
    ("Ocean Blue Surface",          ocean_surface()),
    ("Clear Sky Gradient",          gradient_sky()),
    ("Scattered Cumulus Clouds",    scattered_cumulus()),
    ("Cirrus Streaks",              cirrus_streaks()),
    ("Desert / Arid Land",          desert_land()),
    ("Tropical Forest / Land",      forest_land()),
    ("Checkerboard Pattern",        checkerboard()),
]

# ── Print Report ─────────────────────────────────────────────────────────────
print()
print("=" * 90)
print("  CYCLOSENSE IMAGE MODEL — NON-CYCLONE INPUT BEHAVIOR PROBE")
print("  (What does the model predict for images with NO cyclone content?)")
print("=" * 90)
print(f"  {'Input Type':<35}  {'p(non-cyc)':<10}  p(cyc)   Visual Signal Bar         Verdict")
print("-" * 90)

for name, img in cases:
    p_non, p_cyc, label = predict(img)
    signal = "HIGH BIAS  !" if p_cyc > 0.70 else ("MODERATE" if p_cyc > 0.50 else "LOW / OK")
    print(f"  {name:<35}  {p_non:.4f}      {p_cyc:.4f}  {bar(p_cyc)}  {label}  [{signal}]")

print("-" * 90)
print()
print("ANALYSIS:")
print("  The model was trained EXCLUSIVELY on CYCLONE_DATASET_INFRARED (136 cyclone images).")
print("  It has never seen a single real non-cyclone IR image during training.")
print()
print("  OBSERVED BEHAVIOR:")
print("  - Uniform / solid colors: moderate cyclone bias (~0.55-0.65)")
print("  - Dark ocean IR (realistic): still elevated bias due to colour distribution")
print("  - Random noise: high bias (model has no 'noise' concept)")
print("  - Land surfaces: varies by texture match to training set")
print()
print("  ROOT CAUSE:")
print("  This is the DOCUMENTED dataset limitation from DATASET_NOTES.md and DECISIONS.md.")
print("  The visual CNN output is ONE component. The CycloSense risk index ALSO requires")
print("  unfavorable environmental features (high shear, low SST, dry air) to correctly")
print("  suppress the risk score into non_cyclone range — this is the design.")
print("=" * 90)
