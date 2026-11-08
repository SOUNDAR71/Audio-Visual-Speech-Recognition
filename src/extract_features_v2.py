from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.fft import rfft, dct

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_DIR / "dataset_augmented_v2"
OUTPUT_DIR = PROJECT_DIR / "features_v2"

SAMPLE_RATE = 16000
FRAME_LENGTH = 400
HOP_LENGTH = 160
N_FFT = 512
N_MEL_FILTERS = 40
N_MFCC = 13
PRE_EMPHASIS = 0.97
TARGET_DURATION = 3
SPLITS = ["train", "validation", "test"]


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def create_mel_filterbank():
    mel_points = np.linspace(hz_to_mel(0), hz_to_mel(SAMPLE_RATE / 2), N_MEL_FILTERS + 2)
    hz_points = mel_to_hz(mel_points)
    frequencies = np.linspace(0, SAMPLE_RATE / 2, N_FFT // 2 + 1)
    filters = np.zeros((N_MEL_FILTERS, len(frequencies)), dtype=np.float32)

    for m in range(1, N_MEL_FILTERS + 1):
        left, centre, right = hz_points[m - 1:m + 2]
        for k, freq in enumerate(frequencies):
            if left <= freq <= centre:
                filters[m - 1, k] = (freq - left) / (centre - left)
            elif centre < freq <= right:
                filters[m - 1, k] = (right - freq) / (right - centre)
    return filters


def extract_mfcc(path, filters):
    audio, sr = sf.read(path)

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sr} Hz")

    audio = audio.astype(np.float32)
    target = int(SAMPLE_RATE * TARGET_DURATION)
    audio = audio[:target] if len(audio) > target else np.pad(audio, (0, target - len(audio)))

    audio = np.append(audio[0], audio[1:] - PRE_EMPHASIS * audio[:-1])

    count = 1 + (len(audio) - FRAME_LENGTH) // HOP_LENGTH
    frames = np.array([audio[i * HOP_LENGTH:i * HOP_LENGTH + FRAME_LENGTH] for i in range(count)])
    frames *= np.hamming(FRAME_LENGTH)

    spectrum = np.abs(rfft(frames, n=N_FFT, axis=1)) ** 2 / N_FFT
    mel_energy = np.dot(spectrum, filters.T)
    log_mel = np.log(np.maximum(mel_energy, 1e-10))

    return dct(log_mel, type=2, axis=1, norm="ortho")[:, :N_MFCC].astype(np.float32)


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_DIR}")

    filters = create_mel_filterbank()
    total, failed = 0, []

    for split in SPLITS:
        files = sorted((INPUT_DIR / split).rglob("*.wav"))
        print(f"\nProcessing {split}: {len(files)} files")

        for path in files:
            try:
                features = extract_mfcc(path, filters)
                output = (OUTPUT_DIR / path.relative_to(INPUT_DIR)).with_suffix(".npy")
                output.parent.mkdir(parents=True, exist_ok=True)
                np.save(output, features)
                total += 1
            except Exception as error:
                failed.append(f"{path}: {error}")

        print(f"Completed {split}")

    print("\nMFCC EXTRACTION COMPLETED")
    print("Successful files:", total)
    print("Failed files:", len(failed))
    print("Output folder:", OUTPUT_DIR)

    examples = list(OUTPUT_DIR.rglob("*.npy"))
    if examples:
        print("Example shape:", np.load(examples[0]).shape)

    for error in failed:
        print(error)


if __name__ == "__main__":
    main()
