
from pathlib import Path
from scipy.io.wavfile import read
import sounddevice as sd

dataset_folder = Path("dataset")

names = [
    "Aaron", "Alex", "Ali", "Charlie", "Jacek",
    "Jack", "Jade", "Joshua", "Harriet", "Mason",
    "Nathan", "Nicholas", "Sam", "Samuel", "Sean",
    "Toby", "Tommy", "Xingchen", "Ziang", "Zihan"
]

for name in names:
    recording_number = 1

    while True:
        filename = dataset_folder / f"{name}_{recording_number:02d}.wav"

        if not filename.exists():
            break

        print(f"\nPlaying: {filename}")

        sample_rate, audio = read(str(filename))

        input("Press Enter to play this recording...")
        sd.play(audio, sample_rate)
        sd.wait()

        input("Press Enter to continue to the next recording...")

        recording_number += 1

print("\nFinished playing all recordings!")
