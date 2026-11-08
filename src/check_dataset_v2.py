from pathlib import Path
import soundfile as sf

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset_original_v2"

files = list(DATASET_DIR.rglob("*.wav"))

valid = 0
problems = []

for file in files:
    try:
        audio, sample_rate = sf.read(file)

        if sample_rate != 16000:
            problems.append(f"{file.name}: sample rate is {sample_rate}")

        elif len(audio) == 0:
            problems.append(f"{file.name}: empty audio")

        else:
            valid += 1

    except Exception as error:
        problems.append(f"{file.name}: {error}")

print("Total files:", len(files))
print("Valid recordings:", valid)
print("Problems:", len(problems))

for problem in problems:
    print(problem)
    