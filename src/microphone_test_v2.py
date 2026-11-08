from pathlib import Path
import json
import numpy as np
import sounddevice as sd
import soundfile as sf
import tensorflow as tf
from scipy.fft import rfft, dct

PROJECT_DIR = Path(__file__).resolve().parent.parent
SAMPLE_RATE, DURATION = 16000, 3
FRAME_LENGTH, HOP_LENGTH, N_FFT = 400, 160, 512
N_MEL_FILTERS, N_MFCC = 40, 13
PRE_EMPHASIS = 0.97

MODEL_PATH = PROJECT_DIR / "models_v2/best_cnn_v2.keras"
NORM_PATH = PROJECT_DIR / "models_v2/normalisation_v2.npz"
CLASS_PATH = PROJECT_DIR / "models_v2/class_names_v2.json"
RECORD_PATH = PROJECT_DIR / "results/microphone_recording_v2.wav"

model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_PATH) as f:
    class_names = json.load(f)

norm = np.load(NORM_PATH)
mean_key = next(k for k in norm.files if "mean" in k.lower())
std_key = next(k for k in norm.files if "std" in k.lower() or "standard" in k.lower())
training_mean = norm[mean_key]
training_std = np.where(norm[std_key] == 0, 1, norm[std_key])

print("CNN model loaded successfully!")


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def create_filterbank():
    points = np.linspace(hz_to_mel(0), hz_to_mel(SAMPLE_RATE / 2), N_MEL_FILTERS + 2)
    points = mel_to_hz(points)
    frequencies = np.linspace(0, SAMPLE_RATE / 2, N_FFT // 2 + 1)
    filters = np.zeros((N_MEL_FILTERS, len(frequencies)), dtype=np.float32)

    for m in range(1, N_MEL_FILTERS + 1):
        left, centre, right = points[m - 1:m + 2]
        for k, freq in enumerate(frequencies):
            if left <= freq <= centre:
                filters[m - 1, k] = (freq - left) / (centre - left)
            elif centre < freq <= right:
                filters[m - 1, k] = (right - freq) / (right - centre)
    return filters


def record_audio():
    input("\nPress ENTER to record...")
    print("Speak ONE name now!")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    audio = audio.flatten()
    RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    sf.write(RECORD_PATH, audio, SAMPLE_RATE)
    return audio


def extract_mfcc(audio, filters):
    target = int(SAMPLE_RATE * DURATION)
    audio = audio[:target] if len(audio) > target else np.pad(audio, (0, target - len(audio)))
    audio = np.append(audio[0], audio[1:] - PRE_EMPHASIS * audio[:-1])

    count = 1 + (len(audio) - FRAME_LENGTH) // HOP_LENGTH
    frames = np.array([audio[i * HOP_LENGTH:i * HOP_LENGTH + FRAME_LENGTH] for i in range(count)])
    frames *= np.hamming(FRAME_LENGTH)

    power = np.abs(rfft(frames, n=N_FFT, axis=1)) ** 2 / N_FFT
    log_mel = np.log(np.maximum(np.dot(power, filters.T), 1e-10))
    return dct(log_mel, type=2, axis=1, norm="ortho")[:, :N_MFCC].astype(np.float32)


def main():
    filters = create_filterbank()
    audio = record_audio()
    mfcc = extract_mfcc(audio, filters)

    normalised = (mfcc - training_mean) / training_std
    model_input = np.expand_dims(normalised, axis=(0, -1))

    probabilities = model.predict(model_input, verbose=0)[0]
    index = int(np.argmax(probabilities))

    print("\n========== RECOGNITION RESULT ==========")
    print("Recognised name:", class_names[index])
    print(f"Prediction confidence: {probabilities[index] * 100:.2f}%")
    print("Recording saved:", RECORD_PATH)


if __name__ == "__main__":
    main()
