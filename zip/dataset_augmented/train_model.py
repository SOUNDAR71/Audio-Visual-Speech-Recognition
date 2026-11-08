# from pathlib import Path
# import re
# import numpy as np
# import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# import joblib


# # Project folders
# PROJECT_DIR = Path(__file__).resolve().parent.parent
# FEATURES_DIR = PROJECT_DIR / "features"
# MODELS_DIR = PROJECT_DIR / "models"
# RESULTS_DIR = PROJECT_DIR / "results"

# MODELS_DIR.mkdir(exist_ok=True)
# RESULTS_DIR.mkdir(exist_ok=True)


# # Load feature metadata
# metadata = pd.read_csv(FEATURES_DIR / "metadata.csv")

# X = []
# labels = []

# for _, row in metadata.iterrows():

#     feature_path = PROJECT_DIR / row["feature_file"]

#     mfcc = np.load(feature_path)

#     # Calculate mean and standard deviation for each MFCC coefficient
#     mean_features = np.mean(mfcc, axis=1)
#     std_features = np.std(mfcc, axis=1)

#     # Combine them into 80 features
#     features = np.concatenate([mean_features, std_features])

#     X.append(features)

#     # Convert Aaron_01 into Aaron, Alex_01 into Alex, etc.
#     label = re.sub(r"_\d+$", "", row["name"])
#     labels.append(label)


# X = np.array(X)
# labels = np.array(labels)

# print("Dataset loaded successfully!")
# print("Feature shape:", X.shape)
# print("Number of classes:", len(np.unique(labels)))


# # Encode name labels as numbers
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(labels)


# # Split the dataset
# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.20,
#     random_state=42,
#     stratify=y
# )


# # Standardise the features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)


# # Build and train the DNN
# model = MLPClassifier(
#     hidden_layer_sizes=(128, 64),
#     activation="relu",
#     solver="adam",
#     max_iter=300,
#     random_state=42,
#     early_stopping=True
# )

# print("\nTraining the DNN...")
# model.fit(X_train, y_train)


# # Evaluate the model
# y_pred = model.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)

# print("\nModel evaluation")
# print("----------------")
# print(f"Accuracy: {accuracy * 100:.2f}%")

# print("\nClassification report:")
# print(classification_report(
#     y_test,
#     y_pred,
#     target_names=label_encoder.classes_,
#     zero_division=0
# ))


# # Save the confusion matrix
# cm = confusion_matrix(y_test, y_pred)

# pd.DataFrame(
#     cm,
#     index=label_encoder.classes_,
#     columns=label_encoder.classes_
# ).to_csv(RESULTS_DIR / "confusion_matrix.csv")


# # Save the trained model and preprocessing objects
# joblib.dump(model, MODELS_DIR / "speech_recognition_model.pkl")
# joblib.dump(scaler, MODELS_DIR / "feature_scaler.pkl")
# joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")


# print("\nTraining completed!")
# print("Model saved in:", MODELS_DIR)
# print("Confusion matrix saved in:", RESULTS_DIR)

import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_DIR = Path(__file__).resolve().parent.parent
FEATURE_DIR = PROJECT_DIR / "features"

expected_names = [
    "Aaron", "Alex", "Ali", "Charlie", "Harriet",
    "Jacek", "Jack", "Jade", "Joshua", "Mason",
    "Nathan", "Nicholas", "Sam", "Samuel", "Sean",
    "Toby", "Tommy", "Xingchen", "Ziang", "Zihan"
]

files_by_name = defaultdict(list)

# Find all feature files
for file in FEATURE_DIR.rglob("*.npy"):
    name = file.name.split("_")[0]
    files_by_name[name].append(file)

print("\nMFCC FEATURE CHECK")
print("=" * 55)

total_files = 0

for name in expected_names:
    files = files_by_name[name]
    total_files += len(files)

    if files:
        sample = np.load(files[0])
        print(
            f"{name:12} | Files: {len(files):3} | "
            f"Shape: {sample.shape}"
        )
    else:
        print(f"{name:12} | MISSING")

print("=" * 55)
print("Total feature files:", total_files)
print("Total names:", len(expected_names))
