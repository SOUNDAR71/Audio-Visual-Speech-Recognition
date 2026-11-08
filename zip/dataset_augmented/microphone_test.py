
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import joblib
import re

# Project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Recording settings
SAMPLE_RATE = 16000
DURATION = 3

# MFCC settings (same as training)
N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400

# Model paths
MODEL_PATH = PROJECT_DIR / "models" / "speech_recognition_model.pkl"
SCALER_PATH = PROJECT_DIR / "models" / "feature_scaler.pkl"
ENCODER_PATH = PROJECT_DIR / "models" / "label_encoder.pkl"

# Load trained model
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)


def record_audio():
    print("\nGet ready!")
    input("Press ENTER to start recording...")

    print("Recording... Say ONE name now.")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    audio = audio.flatten()

    recording_path = PROJECT_DIR / "results" / "microphone_recording.wav"
    sf.write(recording_path, audio, SAMPLE_RATE)

    print("Recording saved:", recording_path)

    return audio


def extract_features(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH
    )

    mean_features = np.mean(mfcc, axis=1)
    std_features = np.std(mfcc, axis=1)

    features = np.concatenate(
        [mean_features, std_features]
    )

    return features.reshape(1, -1)


def recognise_name(audio):

    features = extract_features(audio)

    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)

    predicted_name = label_encoder.inverse_transform(
        prediction.astype(int)
    )[0]

    return predicted_name


def main():

    audio = record_audio()

    predicted_name = recognise_name(audio)

    print("\n-----------------------------")
    print(" SPEECH RECOGNITION RESULT")
    print("-----------------------------")
    print("Recognised name:", predicted_name)
    print("-----------------------------")


if __name__ == "__main__":
    main()
