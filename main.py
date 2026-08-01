import numpy as np
import sounddevice as sd

def play_tone(frequency, duration=1.0, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = np.sin(2 * np.pi * frequency * t)
    sd.play(wave, sample_rate)
    sd.wait()

play_tone(440)  # 440 Hz = the note A