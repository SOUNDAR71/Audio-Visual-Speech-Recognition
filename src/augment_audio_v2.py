from pathlib import Path
import csv
import random
import shutil
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

PROJECT_DIR = Path(__file__).resolve().parent.parent
ORIGINAL_DIR = PROJECT_DIR / "dataset_original_v2"
NOISE_DIR = PROJECT_DIR / "noise"
OUTPUT_DIR = PROJECT_DIR / "dataset_augmented_v2"

SAMPLE_RATE = 16000
SNR_LEVELS = [-5, 0, 5, 10, 15]
random.seed(42)
np.random.seed(42)


def load_audio(path):
    audio, sr = sf.read(path)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if sr != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, sr)
    return audio.astype(np.float32)


def add_noise(speech, noise, snr):
    if len(noise) < len(speech):
        noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))))
    start = random.randint(0, len(noise) - len(speech))
    noise = noise[start:start + len(speech)]

    sp = np.mean(speech ** 2)
    npower = np.mean(noise ** 2)
    if sp < 1e-12 or npower < 1e-12:
        raise ValueError("Silent speech or noise recording.")

    target = sp / (10 ** (snr / 10))
    noisy = speech + noise * np.sqrt(target / npower)
    peak = np.max(np.abs(noisy))
    if peak > 0.99:
        noisy *= 0.99 / peak
    return noisy.astype(np.float32)


if not ORIGINAL_DIR.exists() or not NOISE_DIR.exists():
    raise FileNotFoundError("Original dataset or noise folder missing.")

noise_files = list(NOISE_DIR.glob("*.wav")) + list(NOISE_DIR.glob("*.flac"))
name_folders = sorted(f for f in ORIGINAL_DIR.iterdir() if f.is_dir())

if len(name_folders) != 20 or not noise_files:
    raise ValueError("Check the 20 name folders and noise recordings.")

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

for split in ["train", "validation", "test"]:
    (OUTPUT_DIR / split).mkdir(parents=True)

noise_data = {f.name: load_audio(f) for f in noise_files}
noise_data = {k: v for k, v in noise_data.items() if len(v) > 0}
if not noise_data:
    raise ValueError("No usable noise recordings.")

metadata = []

for folder in name_folders:
    name = folder.name
    recordings = sorted(folder.glob("*.wav"))
    if len(recordings) != 20:
        raise ValueError(f"{name}: expected 20 recordings.")

    random.shuffle(recordings)
    train, validation, test = recordings[:12], recordings[12:15], recordings[15:]

    for split, files in [("train", train), ("validation", validation), ("test", test)]:
        out_folder = OUTPUT_DIR / split / name
        out_folder.mkdir(parents=True)

        for source in files:
            output = out_folder / source.name

            if split == "train":
                speech = load_audio(source)
                sf.write(output, speech, SAMPLE_RATE)
                metadata.append([split, name, source.name, output.relative_to(OUTPUT_DIR).as_posix(), "clean"])

                for snr in SNR_LEVELS:
                    noise_name = random.choice(list(noise_data))
                    noisy = add_noise(speech, noise_data[noise_name], snr)
                    noisy_path = out_folder / f"{source.stem}_snr_{snr}dB.wav"
                    sf.write(noisy_path, noisy, SAMPLE_RATE)
                    metadata.append([split, name, source.name, noisy_path.relative_to(OUTPUT_DIR).as_posix(), f"{snr} dB"])
            else:
                shutil.copy2(source, output)
                metadata.append([split, name, source.name, output.relative_to(OUTPUT_DIR).as_posix(), "clean"])

    print(f"{name}: 12 train, 3 validation, 5 test originals")

metadata_path = OUTPUT_DIR / "metadata.csv"
with open(metadata_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["split", "label", "source_recording", "output_file", "snr"])
    writer.writerows(metadata)

print("\nAUDIO AUGMENTATION COMPLETED")
for split in ["train", "validation", "test"]:
    print(f"{split.title()} files:", sum(row[0] == split for row in metadata))
print("Noise recordings:", len(noise_data))
print("Output folder:", OUTPUT_DIR)
print("Metadata saved:", metadata_path)
