
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
FEATURE_DIR = PROJECT_DIR / "features"

# Select one existing MFCC feature file
feature_files = list(FEATURE_DIR.rglob("*.npy"))

if not feature_files:
    raise FileNotFoundError("No MFCC feature files found!")

feature_path = feature_files[0]

# Load the saved MFCC features
mfcc_features = np.load(feature_path)

print("Selected file:", feature_path.name)
print("MFCC shape:", mfcc_features.shape)

# Display MFCC heatmap
plt.figure(figsize=(12, 6))

plt.imshow(
    mfcc_features,
    aspect="auto",
    origin="lower",
    cmap="viridis"
)

plt.colorbar(label="MFCC value")
plt.title(f"MFCC Features - {feature_path.stem}")
plt.xlabel("Time Frames")
plt.ylabel("MFCC Coefficients")

plt.tight_layout()
plt.show()
