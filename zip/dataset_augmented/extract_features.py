
from pathlib import Path
import numpy as np
import pandas as pd
import librosa

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset_augmented"
FEATURE_DIR = PROJECT_DIR / "features"

SAMPLE_RATE = 16000
N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400

def main():
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    total_files = 0

    for name_folder in sorted(DATASET_DIR.iterdir()):
        if not name_folder.is_dir():
            continue

        name = name_folder.name
        output_folder = FEATURE_DIR / name
        output_folder.mkdir(parents=True, exist_ok=True)

        audio_files = sorted(
            file for file in name_folder.iterdir()
            if file.suffix.lower() == ".wav"
        )

        for audio_path in audio_files:
            audio, _ = librosa.load(
                audio_path,
                sr=SAMPLE_RATE,
                mono=True
            )

            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=SAMPLE_RATE,
                n_mfcc=N_MFCC,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                win_length=WIN_LENGTH
            )

            feature_path = output_folder / (
                audio_path.stem + ".npy"
            )

            np.save(feature_path, mfcc)

            records.append({
                "name": name,
                "audio_file": str(audio_path.relative_to(PROJECT_DIR)),
                "feature_file": str(feature_path.relative_to(PROJECT_DIR)),
                "frames": mfcc.shape[1]
            })

            total_files += 1

        print(f"Extracted MFCC features: {name}")

    if not records:
        raise ValueError("No WAV recordings were found.")

    metadata = pd.DataFrame(records)
    metadata.to_csv(FEATURE_DIR / "metadata.csv", index=False)

    print("\nMFCC extraction completed!")
    print(f"Total feature files: {total_files}")
    print(f"Number of names: {metadata['name'].nunique()}")
    print(f"Saved in: {FEATURE_DIR}")

if __name__ == "__main__":
    main()
