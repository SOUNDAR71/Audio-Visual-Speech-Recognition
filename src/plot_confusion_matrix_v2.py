import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = BASE_DIR / "results" / "test_predictions_v2.csv"
CLASS_NAMES_PATH = BASE_DIR / "models_v2" / "class_names_v2.json"
OUTPUT_CSV = BASE_DIR / "results" / "confusion_matrix_v2.csv"
OUTPUT_PNG = BASE_DIR / "results" / "confusion_matrix_v2.png"

# Load predictions
predictions = pd.read_csv(PREDICTIONS_PATH)

# Load the 20 class names in the model's order
with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)

# Calculate confusion matrix
cm = confusion_matrix(
    predictions["Actual"],
    predictions["Predicted"],
    labels=class_names
)

# Save confusion matrix as CSV
cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)
cm_df.to_csv(OUTPUT_CSV)

# Plot confusion matrix
plt.figure(figsize=(14, 12))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix - Improved CNN Model")
plt.colorbar()

positions = range(len(class_names))
plt.xticks(positions, class_names, rotation=90)
plt.yticks(positions, class_names)

plt.xlabel("Predicted Name")
plt.ylabel("Actual Name")

# Display the number of predictions in each cell
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300)
plt.close()

print("Confusion matrix generated successfully!")
print(f"CSV saved: {OUTPUT_CSV}")
print(f"Plot saved: {OUTPUT_PNG}")
