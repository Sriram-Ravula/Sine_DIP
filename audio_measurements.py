import matplotlib.pyplot as plt
import numpy as np
import random

import torch
import torchvision as tv
from torchvision import datasets, transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import math

import time
import wavio

import audio_utils

LR = 5e-4 # learning rate
MOM = 0.9 # momentum
NUM_ITER = 3000 # number iterations
WD = 1e-4 # weight decay for l2-regularization

Z_NUM = 32 # input seed
NGF = 64 # number of filters per layer
BATCH_SIZE = 1 # batch size of gradient step
nc = 1 #num channels in the net I/0


CUDA = torch.cuda.is_available()
print("On GPU: ", CUDA)

#save the correct datatype depending on CPU or GPU execution
if CUDA :
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor


filename = "offer"
test_type = "dropout-16bitscale"


wave_rate, wave_len, wave_res, nc, y0 = audio_utils.read_wav("audio_data/" + filename + "_8192hz_2s.wav")

compressed = True
compressed_noisy = False

if compressed:
    num_measurements = 1000
else:
    num_measurements = wave_len


MU, SIGMA, POWER = audio_utils.get_stats(y0)

y0_normalised = audio_utils.normalise(y0, MU, SIGMA)
y = torch.Tensor(y0_normalised)
y = Variable(y.type(dtype))

spectrum =np.fft.fft(y0_normalised[:,0], norm='ortho')
spectrum = abs(spectrum[0:round(len(spectrum)/2)]) # Just first half of the spectrum, as the second is the negative copy

plt.figure()
plt.plot(spectrum, 'r')
plt.xlabel('Frequency (hz)')
plt.title('Original Waveform')
plt.xlim(0, wave_rate/2)
plt.savefig("Freq_recon/" + filename + "-" + test_type + "/" + filename + "-" + test_type + "_freq.jpg")
plt.close()


#number of measurements to iterate over and record
measurements_list = [100, 200, 300, 500, 1000, 1500, 2000, 2500, 3000, 4000]
#measurements_list = [100, 200, 300, 500, 1000, 2000, 3000, 4000]
mse_list = np.zeros((len(measurements_list), 2))

start = time.time()
for i in range(len(measurements_list)):

    net_A, las_A = audio_utils.get_A(compressed = compressed, noisy = compressed_noisy, num_measurements=measurements_list[i], original_length=wave_len, num_channels=nc)

    meas_lasso = np.matmul(net_A.numpy(), y0_normalised)

    x_hat = audio_utils.run_Lasso(las_A, MU, SIGMA, meas_lasso, filename, test_type, wave_res=wave_res, wave_rate=wave_rate, num_measurements=measurements_list[i], output_size=wave_len, num_channels=nc, alpha= 1e-5)
    x_hat = audio_utils.renormalise(x_hat, MU, SIGMA)

    mse_lasso = np.mean((np.squeeze(x_hat) - np.squeeze(y0))**2)/POWER[0]
    print("\nLasso - " + str(measurements_list[i]) + " :", mse_lasso)

    mse_DIP = audio_utils.run_DIP(filename = filename, test_type=test_type, LR=LR, A=net_A, y=y, y0=y0, dtype=dtype, num_channels=nc, wave_len=wave_len, num_measurements=measurements_list[i], wave_rate = wave_rate, wave_res = wave_res, CUDA=CUDA, num_iter=3000)[-1]
    print("Net - " + str(measurements_list[i]) + " :", mse_DIP)

    mse_list[i,0] = mse_lasso
    mse_list[i,1] = mse_DIP

end = time.time()
print("Execution Time: ", round(end-start, 2), "s")

plt.figure()
plt.plot(measurements_list, mse_list[:, 0], label = "Lasso", color = 'r')
plt.plot(measurements_list, mse_list[:, 1], label = "Net", color = 'b')
plt.xlabel("Num Measurements")
plt.ylabel("MSE")
plt.title(test_type + " Compressed Sensing - Lasso vs. DIP")
plt.legend()
plt.savefig("Freq_recon/" + filename + "-" + test_type + "/" + filename + "-" + test_type + "-" + str(NUM_ITER) + "iter_lasso_net_comp.jpg")
plt.show()






