
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Find the project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"

# Load the confusion matrix
cm = pd.read_csv(
    RESULTS_DIR / "confusion_matrix.csv",
    index_col=0
)

# Create the heatmap
fig, ax = plt.subplots(figsize=(14, 11))

image = ax.imshow(cm.values, cmap="Blues")

# Add name labels
ax.set_xticks(range(len(cm.columns)))
ax.set_yticks(range(len(cm.index)))

ax.set_xticklabels(cm.columns, rotation=45, ha="right")
ax.set_yticklabels(cm.index)

# Add values inside each cell
for i in range(len(cm.index)):
    for j in range(len(cm.columns)):
        ax.text(
            j, i, cm.iloc[i, j],
            ha="center",
            va="center"
        )

ax.set_xlabel("Predicted Name")
ax.set_ylabel("Actual Name")
ax.set_title("Speech Recognition – Confusion Matrix")

fig.colorbar(image, ax=ax, label="Number of recordings")

plt.tight_layout()

# Save the heatmap
output_path = RESULTS_DIR / "confusion_matrix.png"
plt.savefig(output_path, dpi=300)

print("Confusion matrix heatmap saved:", output_path)

plt.show()
