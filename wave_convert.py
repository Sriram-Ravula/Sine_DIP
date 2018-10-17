import numpy as np
import wavio
from scipy import signal

OUTPUT_RATE = 8192
OUTPUT_LENGTH = 2
OUTPUT_CHANNELS = 1

in_filename = "test.wav"
out_filename = "testR.wav"

wav = wavio.read(in_filename)
rate = wav.rate
channels = wav.data.shape[1]
resolution = wav.sampwidth
duration = wav.data.shape[0]/rate

print("Sampling Rate: ", rate)
print("Num Channels: ", channels)
print("Resolution: ", resolution)
print("Length: ", duration)

output_samples = round(OUTPUT_RATE*duration)
resampled_wave = np.zeros((output_samples, OUTPUT_CHANNELS))

for i in range(OUTPUT_CHANNELS):
    resampled_wave[:, i] = signal.resample(x = wav.data[:, i], num = output_samples)


output_wave = np.zeros((OUTPUT_RATE*OUTPUT_LENGTH, OUTPUT_CHANNELS))

if (OUTPUT_RATE*OUTPUT_LENGTH <= resampled_wave.shape[0]):
    output_wave = resampled_wave[0:OUTPUT_RATE*OUTPUT_LENGTH, :]
else:
    output_wave = resampled_wave

wavio.write(out_filename, output_wave, OUTPUT_RATE, sampwidth=resolution)


