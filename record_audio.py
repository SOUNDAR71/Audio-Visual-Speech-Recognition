import sounddevice as sd
from scipy.io.wavfile import write

# Recording settings
duration = 5
sample_rate = 44100

print("Start speaking...")

# Record audio
audio = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,
    dtype='int16'
)

sd.wait()

# Save audio
write("speech.wav", sample_rate, audio)

print("Recording saved successfully!")