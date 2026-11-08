
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report
import joblib

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

# Load saved model components
model = joblib.load(MODELS_DIR / "speech_recognition_model.pkl")
scaler = joblib.load(MODELS_DIR / "feature_scaler.pkl")
label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")

# Load the original test split using the same settings
from sklearn.model_selection import train_test_split
import numpy as np
import re

metadata = pd.read_csv(PROJECT_DIR / "features" / "metadata.csv")

X = []
labels = []

for _, row in metadata.iterrows():
    feature_path = PROJECT_DIR / row["feature_file"]
    mfcc = np.load(feature_path)

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1)
    ])

    X.append(features)
    labels.append(re.sub(r"_\d+$", "", row["name"]))

X = np.array(X)
y = label_encoder.transform(labels)

# Recreate the same test split
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Apply the saved scaler
X_test = scaler.transform(X_test)

# Predict
y_pred = model.predict(X_test)

# Generate classification report
report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_,
    output_dict=True,
    zero_division=0
)

# Save results
report_df = pd.DataFrame(report).transpose()
output_path = RESULTS_DIR / "evaluation_summary.csv"
report_df.to_csv(output_path)

print(report_df.round(3))
print("\nEvaluation metrics saved:", output_path)
