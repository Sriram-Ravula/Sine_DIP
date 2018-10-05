import numpy as np
import wavio
import matplotlib.pyplot as plt

rate = 22050             # samples per second
T = 3                    # sample duration (seconds)
f = 500.0                # sound frequency (Hz)
t = np.linspace(0, T, T*rate, endpoint=False)
sig = np.sin(2 * np.pi * f * t)
sig = sig + np.sin(2 * np.pi * 10*f * t)

wavio.write("sine24.wav", sig, rate, sampwidth=3)

wav = wavio.read("guitar.wav")
print("rate: ", wav.rate)
print("length (s): ", round(wav.data.shape[0]/(1.0*wav.rate), 4))
print("bits/sample: ", 8*wav.sampwidth)
#wavio.write("sine24.wav", wav.data, wav.rate, sampwidth=2)