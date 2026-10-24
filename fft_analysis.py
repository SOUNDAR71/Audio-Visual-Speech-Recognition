import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

# Read speech audio
sample_rate, audio = read("speech.wav")

# Convert stereo audio to mono
if audio.ndim > 1:
    audio = audio[:, 0]

# Frame settings
frame_length = 320

# 50% overlap
overlap = 160
hop_length = frame_length - overlap

# Calculate number of overlapping frames
num_frames = 1 + (len(audio) - frame_length) // hop_length

print("Sample rate:", sample_rate)
print("Total samples:", len(audio))
print("Frame length:", frame_length)
print("Overlap:", overlap)
print("Hop length:", hop_length)
print("Number of frames:", num_frames)

# Hamming window
window = np.hamming(frame_length)

# Store magnitude spectra
spectra = []

# Process every overlapping frame
for i in range(num_frames):

    # Find start and end of frame
    start = i * hop_length
    end = start + frame_length

    # Extract frame
    frame = audio[start:end]

    # Apply Hamming window
    frame = frame * window

    # Calculate FFT
    fft_result = np.fft.fft(frame)

    # Calculate magnitude
    magnitude = np.abs(fft_result)

    # Keep positive frequencies
    magnitude = magnitude[:frame_length // 2]

    # Store magnitude
    spectra.append(magnitude)

# Convert list to NumPy array
spectra = np.array(spectra)

# Frequency axis
frequencies = np.fft.fftfreq(
    frame_length,
    1 / sample_rate
)[:frame_length // 2]

# Time axis using hop length
times = np.arange(num_frames) * hop_length / sample_rate

# Create spectrogram
plt.figure(figsize=(12, 5))

plt.imshow(
    spectra.T,
    aspect="auto",
    origin="lower",
    extent=[
        times[0],
        times[-1],
        frequencies[0],
        frequencies[-1]
    ]
)

plt.title("Short-Time Magnitude Spectrum with Overlapping Frames")
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")
plt.ylim(0, 5000)
plt.colorbar(label="Magnitude")

plt.show()