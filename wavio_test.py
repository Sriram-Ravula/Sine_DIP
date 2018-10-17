import numpy as np
import wavio
import matplotlib.pyplot as plt

rate = 8192             # samples per second
T = 2                    # sample duration (seconds)
f = 350.0                # sound frequency (Hz)
t = np.linspace(0, T, T*rate, endpoint=False)
sig = np.sin(2 * np.pi * f * t)
sig = sig + np.sin(2 * np.pi * 4*f * t)

wav = wavio.read("audio_data/testR.wav")
print("rate: ", wav.rate)
print("length: ", wav.data.shape[0])
print("bits/sample: ", 8*wav.sampwidth)
print("MAX: ", np.max(wav.data[:, 0]))

wav = wavio.read("audio_data/test.wav")
print("\nrate: ", wav.rate)
print("length: ", wav.data.shape[0])
print("bits/sample: ", 8*wav.sampwidth)
print("MAX: ", np.max(wav.data[:, 0]))

spectrum =np.fft.fft(wav.data[:,0])
spectrum = abs(spectrum[0:round(len(spectrum)/2)]) # Just first half of the spectrum, as the second is the negative copy
plt.figure()
plt.plot(spectrum, 'r')
plt.xlabel('Frequency (hz)')
plt.title('Original Waveform')
plt.xlim(0,wav.rate/2)
plt.show()

#wave = np.zeros((len(sig), 2))
#wave[:, 0] = sig
#wave[:, 1] = np.sin(2 * np.pi * 8*f * t)

#wavio.write("sine24.wav", wave, rate, sampwidth=2)