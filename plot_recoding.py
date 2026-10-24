
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from pathlib import Path

project_folder = Path(__file__).resolve().parent

audio_file = project_folder / "dataset" / "Aaron_01.wav"

sample_rate, audio = read(audio_file)

if audio.ndim == 2:
    audio = audio.mean(axis=1)

time = np.arange(len(audio)) / sample_rate

plt.figure(figsize=(12, 4))
plt.plot(time, audio)
plt.title("Waveform of Aaron")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()
