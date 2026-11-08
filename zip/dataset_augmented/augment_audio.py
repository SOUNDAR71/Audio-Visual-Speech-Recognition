
from pathlib import Path
import random
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent

VOICE_DIR = PROJECT_DIR / "dataset_original"
NOISE_DIR = PROJECT_DIR / "noise"
OUTPUT_DIR = PROJECT_DIR / "dataset_augmented"

# Dataset settings
TARGET_SAMPLES_PER_NAME = 20
SNR_LEVELS = [-5, 0, 5, 10, 15]

random.seed(42)


def convert_to_mono(audio):
    """Convert stereo audio into mono."""
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    return audio.astype(np.float32)


def match_sample_rate(audio, original_rate, target_rate):
    """Resample noise to match the speech sample rate."""
    if original_rate == target_rate:
        return audio

    common = gcd(original_rate, target_rate)

    return resample_poly(
        audio,
        target_rate // common,
        original_rate // common
    ).astype(np.float32)


def prepare_noise(noise, speech_length):
    """Make noise the same length as speech."""
    if len(noise) < speech_length:
        repeats = int(np.ceil(speech_length / len(noise)))
        noise = np.tile(noise, repeats)

    if len(noise) > speech_length:
        start = random.randint(0, len(noise) - speech_length)
        noise = noise[start:start + speech_length]

    return noise


def mix_at_snr(speech, noise, snr_db):
    """Mix speech and noise at the requested SNR."""

    speech_power = np.mean(speech ** 2)
    noise_power = np.mean(noise ** 2)

    if speech_power <= 1e-12 or noise_power <= 1e-12:
        raise ValueError("Speech or noise contains almost no audio energy.")

    noise_scale = np.sqrt(
        speech_power / (noise_power * (10 ** (snr_db / 10)))
    )

    mixed_audio = speech + noise_scale * noise

    # Prevent clipping while preserving the mixture
    peak = np.max(np.abs(mixed_audio))
    if peak > 1.0:
        mixed_audio = mixed_audio / peak

    return mixed_audio.astype(np.float32)


def main():

    voice_files = sorted(VOICE_DIR.glob("*.wav"))
    noise_files = sorted(NOISE_DIR.glob("*.wav"))

    if not voice_files:
        raise FileNotFoundError(
            f"No WAV voice recordings found in {VOICE_DIR}"
        )

    if not noise_files:
        raise FileNotFoundError(
            f"No WAV noise recordings found in {NOISE_DIR}"
        )

    if len(voice_files) != 20:
        raise ValueError(
            f"Expected 20 original voice files, found {len(voice_files)}."
        )

    print(f"Voice recordings found: {len(voice_files)}")
    print(f"Noise recordings found: {len(noise_files)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_created = 0

    for voice_index, voice_path in enumerate(voice_files):

        speech, speech_rate = sf.read(
            voice_path, dtype="float32"
        )

        speech = convert_to_mono(speech)

        name = voice_path.stem
        name_output_dir = OUTPUT_DIR / name
        name_output_dir.mkdir(parents=True, exist_ok=True)

        # Save original clean recording
        sf.write(
            name_output_dir / f"{name}_clean.wav",
            speech,
            speech_rate,
            subtype="PCM_16"
        )

        print(f"\nProcessing: {name}")

        # Generate 19 noisy versions
        for version in range(TARGET_SAMPLES_PER_NAME - 1):

            noise_path = noise_files[
                (voice_index * (TARGET_SAMPLES_PER_NAME - 1)
                 + version) % len(noise_files)
            ]

            noise, noise_rate = sf.read(
                noise_path, dtype="float32"
            )

            noise = convert_to_mono(noise)

            noise = match_sample_rate(
                noise, noise_rate, speech_rate
            )

            noise = prepare_noise(noise, len(speech))

            snr_db = SNR_LEVELS[
                version % len(SNR_LEVELS)
            ]

            augmented = mix_at_snr(
                speech, noise, snr_db
            )

            output_name = (
                f"{name}_noise_{version + 1:02d}"
                f"_snr_{snr_db}dB.wav"
            )

            sf.write(
                name_output_dir / output_name,
                augmented,
                speech_rate,
                subtype="PCM_16"
            )

            total_created += 1

        print(f"Completed: {name} (20 recordings)")

    print("\nDataset augmentation completed!")
    print(f"Original recordings: {len(voice_files)}")
    print(f"Noisy recordings: {total_created}")
    print(f"Total recordings: {len(voice_files) + total_created}")
    print(f"Saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
