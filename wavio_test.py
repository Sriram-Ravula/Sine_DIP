import numpy as np
import wavio
import matplotlib.pyplot as plt

rate = 8192             # samples per second
T = 2                    # sample duration (seconds)
f = 500.0                # sound frequency (Hz)
t = np.linspace(0, T, T*rate, endpoint=False)
sig = np.sin(2 * np.pi * f * t)
sig = sig + np.sin(2 * np.pi * 4*f * t)

wav = wavio.read("lowres.wav")
print("rate: ", wav.rate)
print("length (s): ", round(wav.data.shape[0]/(1.0*wav.rate), 4))
print("bits/sample: ", 8*wav.sampwidth)
print(np.min(wav.data[:, 0]))

wav = wavio.read("sine24.wav")
print("rate: ", wav.rate)
print("length: ", wav.data.shape[0])
print("bits/sample: ", 8*wav.sampwidth)
print(np.max(wav.data[:, 0]))

wavio.write("sine24.wav", sig, rate, sampwidth=2)