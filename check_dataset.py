
from pathlib import Path
from collections import Counter
from scipy.io.wavfile import read

dataset_folder = Path("dataset")

names = [
    "Aaron", "Alex", "Ali", "Charlie", "Jacek",
    "Jack", "Jade", "Joshua", "Harriet", "Mason",
    "Nathan", "Nicholas", "Sam", "Samuel", "Sean",
    "Toby", "Tommy", "Xingchen", "Ziang", "Zihan"
]

if not dataset_folder.exists():
    print("ERROR: The dataset folder was not found.")
else:
    files = list(dataset_folder.glob("*.wav"))
    counts = Counter()

    print("DATASET CHECK")
    print("-" * 40)

    for file in files:
        name = file.stem.rsplit("_", 1)[0]

        try:
            sample_rate, audio = read(file)
            duration = len(audio) / sample_rate

            if name in names:
                counts[name] += 1

            print(
                f"{file.name}: "
                f"{sample_rate} Hz, "
                f"{duration:.2f} seconds"
            )

        except Exception as error:
            print(f"Could not read {file.name}: {error}")

    print("\nRECORDING COUNT PER NAME")
    print("-" * 40)

    for name in names:
        print(f"{name}: {counts[name]} recording(s)")

    print(f"\nTotal WAV files found: {len(files)}")

