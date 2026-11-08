
import pandas as pd
import numpy as np
import joblib
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Project folders
metadata_path = "features/metadata.csv"
model_path = "models/speech_recognition_model.pkl"
scaler_path = "models/feature_scaler.pkl"
encoder_path = "models/label_encoder.pkl"

# Load metadata
df = pd.read_csv(metadata_path)

# Load trained model components
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(encoder_path)

# Load MFCC features
X = []
y = []

for _, row in df.iterrows():
    features = np.load(row["feature_file"])

    # Calculate mean and standard deviation
    mean_features = np.mean(features, axis=1)
    std_features = np.std(features, axis=1)

    combined_features = np.concatenate(
        [mean_features, std_features]
    )

    X.append(combined_features)

    # Remove recording suffix, e.g. Aaron_01 -> Aaron
    name = re.sub(r"_\d+$", "", row["name"])
    y.append(name)

X = np.array(X)
y = np.array(y)

# Recreate the original 80/20 split
indices = np.arange(len(y))

_, test_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Select only the original test recordings
X_test = X[test_indices]
y_test = y[test_indices]

test_metadata = df.iloc[test_indices].copy()

# Scale features and predict
X_test_scaled = scaler.transform(X_test)
predictions = model.predict(X_test_scaled)
predictions = label_encoder.inverse_transform(predictions.astype(int))

# Extract SNR levels from filenames
test_metadata["snr"] = test_metadata["audio_file"].str.extract(
    r"snr_(-?\d+)dB",
    expand=False
)

test_metadata["snr"] = test_metadata["snr"].fillna("Clean")

# Calculate accuracy for each SNR level
results = []

for snr_level in ["Clean", "-5", "0", "5", "10", "15"]:

    mask = test_metadata["snr"].astype(str).to_numpy() == snr_level

    if np.sum(mask) == 0:
        continue

    actual = y_test[mask]
    predicted = predictions[mask]

    accuracy = accuracy_score(actual, predicted)

    results.append({
        "SNR": snr_level,
        "Test recordings": int(np.sum(mask)),
        "Correct predictions": int(np.sum(actual == predicted)),
        "Accuracy (%)": round(accuracy * 100, 2)
    })

# Save results
results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/noise_evaluation.csv",
    index=False
)

print("\nNOISE EVALUATION RESULTS")
print("========================")
print(results_df.to_string(index=False))

print("\nResults saved to results/noise_evaluation.csv")
