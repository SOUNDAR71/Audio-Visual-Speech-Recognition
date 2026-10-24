import sounddevice as sd
from scipy.io.wavfile import write
from pathlib import Path

# saving theaudio file

dataset_floder = Path("dataset")

sample_rate = 16000
duration = 6


#The names 
names = [
    "Aaron",
    "Alex",
    "Ali",
    "Charlie",
    "Jacek",
    "Jack",
    "Jade",
    "Joshua",
    "Harriet",
    "Mason",
    "Nathan",
    "Nicholas",
    "Sam",
    "Samuel",
    "Sean",
    "Toby",
    "Tommy",
    "Xingchen",
    "Ziang",
    "Zihan" 
]
#Recordind name

for name in names:

    recording_number = 1

    while True:
        filename = dataset_floder/ (f"{name}_{recording_number:02d}.wav")

        if not filename.exists():
            break
        recording_number += 1

    

    # exitsting_file = list(dataset_floder.flob(f"{name}_*.wav"))
    # recording_number = len(exitsing_file) + 1


    filename = dataset_floder / f"{name}_{recording_number:02d}.wav"

    input(f"\n press Enter to record the  {name}....")

    print(f"Record {name} for {duration} seconds. speak now!")

    audio = sd.rec(int (duration * sample_rate), samplerate=sample_rate,channels=1, dtype="int16" )

    sd.wait()
    write(str(filename), sample_rate, audio)

    print(f"Recording saved as {filename}")

    print(f"saved:{filename} \n")
print("All recordings completed!")

