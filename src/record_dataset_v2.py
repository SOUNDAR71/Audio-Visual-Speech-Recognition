import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
import time

PROJECT_DIR = Path(__file__).resolve().parent.parent
SAVE_DIR = PROJECT_DIR / "dataset_original_v2"

SAMPLE_RATE = 16000
DURATION = 3
RECORDINGS_PER_NAME = 20

names = [
    "Aaron", "Alex", "Ali", "Charlie", "Harriet",
    "Jacek", "Jack", "Jade", "Joshua", "Mason",
    "Nathan", "Nicholas", "Sam", "Samuel", "Sean",
    "Toby", "Tommy", "Xingchen", "Ziang", "Zihan"
]

SAVE_DIR.mkdir(parents=True, exist_ok=True)

print("SPEECH DATASET RECORDING SYSTEM")
print("Total recordings:", len(names) * RECORDINGS_PER_NAME)
print("Sample rate:", SAMPLE_RATE, "Hz")
print("Duration:", DURATION, "seconds")
time.sleep(1)

for name in names:
    folder = SAVE_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    print(f"\nRecording name: {name}")

    for i in range(1, RECORDINGS_PER_NAME + 1):
        path = folder / f"{name}_{i:02d}.wav"
        if path.exists():
            print("Already exists:", path.name)
            continue

        while True:
            command = input(f"Press Enter to record {name}, r to skip, q to quit: ").lower()
            if command == "q":
                print("Recording stopped. Files are saved.")
                raise SystemExit
            if command == "r":
                break

            time.sleep(1)
            print("Speak now!")
            try:
                audio = sd.rec(int(DURATION * SAMPLE_RATE),
                               samplerate=SAMPLE_RATE, channels=1,
                               dtype="float32")
                sd.wait()

                if np.max(np.abs(audio)) < 0.005:
                    print("Too quiet. Try again.")
                    continue

                if input("Save recording? (y/r): ").lower() == "y":
                    sf.write(path, audio, SAMPLE_RATE)
                    print("Saved:", path.name)
                    break
            except Exception as error:
                print("Recording error:", error)
                input("Check microphone and press Enter to retry.")

print("\nRecording session completed!")
print("Dataset saved at:", SAVE_DIR)
