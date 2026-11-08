from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR, MODEL_DIR = ROOT / "features_v2", ROOT / "models_v2"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load model, class names and normalisation values
model = tf.keras.models.load_model(MODEL_DIR / "best_cnn_v2.keras")
with open(MODEL_DIR / "class_names_v2.json", encoding="utf-8") as f:
    names = json.load(f)
norm = np.load(MODEL_DIR / "normalisation_v2.npz")
mean, std = norm["mean"], norm["std"]
std = np.where(std == 0, 1, std)

# Load untouched test features
X, y = [], []
for label, name in enumerate(names):
    for file in sorted((FEATURE_DIR / "test" / name).glob("*.npy")):
        feature = np.load(file)
        if feature.shape != (298, 13):
            raise ValueError(f"Invalid feature shape: {file}")
        X.append(feature)
        y.append(label)

X, y = np.asarray(X, np.float32), np.asarray(y, np.int32)
X = ((X[..., np.newaxis] - mean[None, None, :, None]) /
     std[None, None, :, None])

print("Test features:", X.shape)
print("Test recordings:", len(y))

# Predict test recordings
probabilities = model.predict(X, verbose=1)
predictions = np.argmax(probabilities, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y, predictions)
correct = int(np.sum(y == predictions))
print(f"\nCorrect: {correct}/{len(y)}")
print(f"Incorrect: {len(y) - correct}/{len(y)}")
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Generate classification report and confusion matrix
report = classification_report(
    y, predictions, labels=np.arange(len(names)),
    target_names=names, zero_division=0
)
print("\nCLASSIFICATION REPORT\n", report)

cm = confusion_matrix(y, predictions, labels=np.arange(len(names)))
pd.DataFrame(cm, index=names, columns=names).to_csv(
    RESULTS_DIR / "confusion_matrix_v2.csv"
)

# Save reports and individual predictions
(RESULTS_DIR / "classification_report_v2.txt").write_text(
    report, encoding="utf-8"
)
pd.DataFrame({
    "Actual": [names[i] for i in y],
    "Predicted": [names[i] for i in predictions],
    "Correct": y == predictions
}).to_csv(RESULTS_DIR / "test_predictions_v2.csv", index=False)

print("\nEvaluation completed!")
print("Results saved in:", RESULTS_DIR)
