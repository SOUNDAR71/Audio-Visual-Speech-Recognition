from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# 1. Project settings
PROJECT_DIR = Path(__file__).resolve().parent.parent
FEATURE_DIR = PROJECT_DIR / "features_v2"
MODEL_DIR = PROJECT_DIR / "models_v2"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 30
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# 2. Load MFCC features and labels
def load_dataset(split):
    split_dir = FEATURE_DIR / split

    if not split_dir.exists():
        raise FileNotFoundError(f"Missing folder: {split_dir}")

    class_names = sorted(
        p.name for p in split_dir.iterdir() if p.is_dir()
    )

    features, labels = [], []

    for label, name in enumerate(class_names):
        for file in sorted((split_dir / name).glob("*.npy")):
            feature = np.load(file)

            if feature.shape != (298, 13):
                raise ValueError(f"Invalid feature shape: {file}")

            features.append(feature)
            labels.append(label)

    X = np.asarray(features, dtype=np.float32)[..., np.newaxis]
    y = np.asarray(labels, dtype=np.int32)

    return X, y, class_names


# 3. Load training and validation data
X_train, y_train, class_names = load_dataset("train")
X_val, y_val, val_classes = load_dataset("validation")

if class_names != val_classes or len(class_names) != 20:
    raise ValueError("Training and validation classes do not match.")

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Classes:", len(class_names))


# 4. Normalise features using training data only
mean = X_train.mean(axis=(0, 1, 3))
std = np.maximum(X_train.std(axis=(0, 1, 3)), 1e-6)

X_train = (X_train - mean[None, None, :, None]) / std[None, None, :, None]
X_val = (X_val - mean[None, None, :, None]) / std[None, None, :, None]

np.savez(MODEL_DIR / "normalisation_v2.npz", mean=mean, std=std)


# 5. Build the 2D CNN
model = models.Sequential([
    layers.Input(shape=(298, 13, 1)),

    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),

    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(len(class_names), activation="softmax")
])


# 6. Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# 7. Training callbacks
training_callbacks = [
    callbacks.EarlyStopping(
        monitor="val_loss", patience=7, restore_best_weights=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=0.00001
    ),
    callbacks.ModelCheckpoint(
        MODEL_DIR / "best_cnn_v2.keras",
        monitor="val_accuracy", save_best_only=True, mode="max"
    )
]


# 8. Train the CNN
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=training_callbacks,
    shuffle=True
)


# 9. Save model and class names
model.save(MODEL_DIR / "final_cnn_v2.keras")

with open(MODEL_DIR / "class_names_v2.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=4)


# 10. Display training results
best_epoch = np.argmax(history.history["val_accuracy"]) + 1
best_accuracy = max(history.history["val_accuracy"])

print("\nCNN TRAINING COMPLETED")
print("Best epoch:", best_epoch)
print(f"Best validation accuracy: {best_accuracy * 100:.2f}%")
print(f"Final training accuracy: {history.history['accuracy'][-1] * 100:.2f}%")

print("\nSaved files:")
print(MODEL_DIR / "best_cnn_v2.keras")
print(MODEL_DIR / "final_cnn_v2.keras")
print(MODEL_DIR / "normalisation_v2.npz")
print(MODEL_DIR / "class_names_v2.json")
