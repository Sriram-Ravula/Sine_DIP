import numpy as np
import wavio
import matplotlib.pyplot as plt

rate = 8192             # samples per second
T = 2                    # sample duration (seconds)
f = 350.0                # sound frequency (Hz)
t = np.linspace(0, T, T*rate, endpoint=False)
sig = np.sin(2 * np.pi * f * t)
sig = sig + np.sin(2 * np.pi * 4*f * t)

wav = wavio.read("sine24.wav")
print("rate: ", wav.rate)
print("length: ", wav.data.shape[0])
print("bits/sample: ", 8*wav.sampwidth)
print(np.max(wav.data[:, 0]))

wave = np.zeros((len(sig), 2))
wave[:, 0] = sig
wave[:, 1] = np.sin(2 * np.pi * 8*f * t)

wavio.write("sine24.wav", wave, rate, sampwidth=2)