from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

from scipy.signal import resample_poly
from scipy.fft import rfft, dct
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# 1. Project paths and settings
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

ORIGINAL_DIR = PROJECT_DIR / "dataset_original_v2"
NOISE_DIR = PROJECT_DIR / "noise"
DATASET_DIR = PROJECT_DIR / "dataset_augmented_v2"
MODEL_DIR = PROJECT_DIR / "models_v2"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
TARGET_DURATION = 3.0

FRAME_LENGTH = 400
HOP_LENGTH = 160
N_FFT = 512

N_MEL_FILTERS = 40
N_MFCC = 13

PRE_EMPHASIS = 0.97

SNR_LEVELS = [-5, 0, 5, 10, 15]

random.seed(42)
np.random.seed(42)


# --------------------------------------------------
# 2. Load audio
# --------------------------------------------------

def load_audio(file_path):

    audio, sr = sf.read(file_path)

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    if sr != SAMPLE_RATE:
        audio = resample_poly(
            audio,
            SAMPLE_RATE,
            sr
        )

    return audio.astype(np.float32)


# --------------------------------------------------
# 3. Add noise at a specified SNR
# --------------------------------------------------

def add_noise(clean_audio, noise_audio, snr_db):

    if len(noise_audio) < len(clean_audio):

        repeats = int(np.ceil(
            len(clean_audio) / len(noise_audio)
        ))

        noise_audio = np.tile(noise_audio, repeats)

    start = random.randint(
        0,
        len(noise_audio) - len(clean_audio)
    )

    noise = noise_audio[
        start:start + len(clean_audio)
    ]

    speech_power = np.mean(clean_audio ** 2)
    noise_power = np.mean(noise ** 2)

    if speech_power < 1e-12 or noise_power < 1e-12:
        raise ValueError("Speech or noise is silent.")

    target_noise_power = (
        speech_power / (10 ** (snr_db / 10))
    )

    scale = np.sqrt(
        target_noise_power / noise_power
    )

    noisy_audio = clean_audio + scale * noise

    peak = np.max(np.abs(noisy_audio))

    if peak > 0.99:
        noisy_audio = noisy_audio * (0.99 / peak)

    return noisy_audio.astype(np.float32)


# --------------------------------------------------
# 4. Create Mel filterbank
# --------------------------------------------------

def hz_to_mel(frequency):

    return 2595 * np.log10(1 + frequency / 700)


def mel_to_hz(mel):

    return 700 * (10 ** (mel / 2595) - 1)


def create_mel_filterbank():

    minimum_mel = hz_to_mel(0)
    maximum_mel = hz_to_mel(SAMPLE_RATE / 2)

    mel_points = np.linspace(
        minimum_mel,
        maximum_mel,
        N_MEL_FILTERS + 2
    )

    hz_points = mel_to_hz(mel_points)

    fft_frequencies = np.linspace(
        0,
        SAMPLE_RATE / 2,
        N_FFT // 2 + 1
    )

    filterbank = np.zeros(
        (N_MEL_FILTERS, len(fft_frequencies)),
        dtype=np.float32
    )

    for m in range(1, N_MEL_FILTERS + 1):

        left = hz_points[m - 1]
        centre = hz_points[m]
        right = hz_points[m + 1]

        for k, frequency in enumerate(fft_frequencies):

            if left <= frequency <= centre:

                filterbank[m - 1, k] = (
                    (frequency - left) / (centre - left)
                )

            elif centre < frequency <= right:

                filterbank[m - 1, k] = (
                    (right - frequency) / (right - centre)
                )

    return filterbank


# --------------------------------------------------
# 5. Extract MFCC from an audio signal
# --------------------------------------------------

def extract_mfcc(audio, filterbank):

    target_samples = int(
        SAMPLE_RATE * TARGET_DURATION
    )

    if len(audio) > target_samples:

        audio = audio[:target_samples]

    elif len(audio) < target_samples:

        audio = np.pad(
            audio,
            (0, target_samples - len(audio))
        )

    # Pre-emphasis
    emphasized = np.append(
        audio[0],
        audio[1:] - PRE_EMPHASIS * audio[:-1]
    )

    # Framing
    number_of_frames = (
        1 + (len(emphasized) - FRAME_LENGTH) // HOP_LENGTH
    )

    frames = np.zeros(
        (number_of_frames, FRAME_LENGTH),
        dtype=np.float32
    )

    for i in range(number_of_frames):

        start = i * HOP_LENGTH

        frames[i] = emphasized[
            start:start + FRAME_LENGTH
        ]

    # Hamming window
    frames *= np.hamming(FRAME_LENGTH)

    # Power spectrum
    fft_result = rfft(
        frames,
        n=N_FFT,
        axis=1
    )

    power_spectrum = (
        np.abs(fft_result) ** 2
    ) / N_FFT

    # Mel filterbank
    mel_energies = np.dot(
        power_spectrum,
        filterbank.T
    )

    mel_energies = np.maximum(
        mel_energies,
        1e-10
    )

    log_mel = np.log(mel_energies)

    # DCT
    mfcc = dct(
        log_mel,
        type=2,
        axis=1,
        norm="ortho"
    )

    return mfcc[:, :N_MFCC].astype(np.float32)


# --------------------------------------------------
# 6. Load model and supporting files
# --------------------------------------------------

print("Loading trained CNN...")

model = tf.keras.models.load_model(
    MODEL_DIR / "best_cnn_v2.keras"
)

with open(
    MODEL_DIR / "class_names_v2.json",
    "r",
    encoding="utf-8"
) as file:

    class_names = json.load(file)

normalisation = np.load(
    MODEL_DIR / "normalisation_v2.npz"
)

feature_mean = normalisation["mean"]
feature_std = normalisation["std"]

filterbank = create_mel_filterbank()

print("CNN loaded successfully.")
print("Number of classes:", len(class_names))


# --------------------------------------------------
# 7. Load test metadata
# --------------------------------------------------

metadata = pd.read_csv(
    DATASET_DIR / "metadata.csv"
)

test_metadata = metadata[
    metadata["split"] == "test"
].copy()

if len(test_metadata) != 100:

    raise ValueError(
        f"Expected 100 test recordings, found {len(test_metadata)}"
    )

for class_name in class_names:

    count = (
        test_metadata["label"] == class_name
    ).sum()

    if count != 5:

        raise ValueError(
            f"{class_name}: expected 5 test recordings, found {count}"
        )

print("\n100 held-out test recordings confirmed.")


# --------------------------------------------------
# 8. Load noise recordings
# --------------------------------------------------

noise_files = sorted(
    list(NOISE_DIR.glob("*.wav")) +
    list(NOISE_DIR.glob("*.flac"))
)

if not noise_files:

    raise FileNotFoundError(
        "No WAV or FLAC noise recordings found."
    )

noise_data = {}

for noise_file in noise_files:

    noise_data[noise_file.name] = load_audio(
        noise_file
    )

print("Noise recordings loaded:", len(noise_data))


# --------------------------------------------------
# 9. Evaluate clean and noisy test recordings
# --------------------------------------------------

all_results = []
all_predictions = []

test_conditions = ["Clean"] + SNR_LEVELS

for condition in test_conditions:

    print("\n====================================")
    print("Evaluating:", condition)
    print("====================================")

    X = []
    y_true = []
    noise_names = []

    for _, row in test_metadata.iterrows():

        label = row["label"]

        source_path = (
            ORIGINAL_DIR /
            label /
            row["source_recording"]
        )

        if not source_path.exists():

            raise FileNotFoundError(
                f"Test recording not found: {source_path}"
            )

        clean_audio = load_audio(source_path)

        if condition == "Clean":

            test_audio = clean_audio
            selected_noise = "None"

        else:

            selected_noise = random.choice(
                list(noise_data.keys())
            )

            test_audio = add_noise(
                clean_audio,
                noise_data[selected_noise],
                condition
            )

        # Manual MFCC extraction
        mfcc = extract_mfcc(
            test_audio,
            filterbank
        )

        if mfcc.shape != (298, 13):

            raise ValueError(
                f"Unexpected MFCC shape: {mfcc.shape}"
            )

        X.append(mfcc)
        y_true.append(class_names.index(label))
        noise_names.append(selected_noise)

    # Convert to NumPy
    X = np.asarray(X, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.int32)

    # Add CNN channel dimension
    X = X[..., np.newaxis]

    # Apply saved training normalisation
    X = (
        X - feature_mean[None, None, :, None]
    ) / feature_std[None, None, :, None]

    # Predict
    probabilities = model.predict(
        X,
        verbose=0
    )

    y_pred = np.argmax(
        probabilities,
        axis=1
    )

    # Calculate accuracy
    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    correct = int(
        np.sum(y_true == y_pred)
    )

    total = len(y_true)

    print("Correct predictions:", correct, "/", total)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    all_results.append({
        "Condition": (
            "Clean" if condition == "Clean"
            else f"{condition} dB"
        ),
        "SNR_dB": (
            "Clean" if condition == "Clean"
            else condition
        ),
        "Correct": correct,
        "Total": total,
        "Accuracy_Percent": round(
            accuracy * 100, 2
        )
    })

    # Save individual predictions
    for i in range(total):

        all_predictions.append({
            "Condition": (
                "Clean" if condition == "Clean"
                else f"{condition} dB"
            ),
            "Actual": class_names[y_true[i]],
            "Predicted": class_names[y_pred[i]],
            "Correct": bool(y_true[i] == y_pred[i]),
            "Noise_File": noise_names[i]
        })


# --------------------------------------------------
# 10. Save final results
# --------------------------------------------------

results_df = pd.DataFrame(all_results)

predictions_df = pd.DataFrame(all_predictions)

results_df.to_csv(
    RESULTS_DIR / "noise_robustness_v2.csv",
    index=False
)

predictions_df.to_csv(
    RESULTS_DIR / "noise_predictions_v2.csv",
    index=False
)


# --------------------------------------------------
# 11. Display final summary
# --------------------------------------------------

print("\n============================================")
print("FINAL NOISE ROBUSTNESS EVALUATION")
print("============================================")

print(
    results_df.to_string(index=False)
)

print("\nResults saved:")
print(
    RESULTS_DIR / "noise_robustness_v2.csv"
)

print(
    RESULTS_DIR / "noise_predictions_v2.csv"
)

print("\nNoise robustness evaluation completed!")
