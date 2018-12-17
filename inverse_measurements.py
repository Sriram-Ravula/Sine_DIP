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

import inverse_utils
import dip_utils

LR = 1e-4 # learning rate
MOM = 0.9 # momentum
NUM_ITER = 3000 # number iterations
WD = 1 # weight decay for l2-regularization
Z_NUM = 32 # input seed
NGF = 64 # number of filters per layer
BATCH_SIZE = 1 # batch size of gradient step
nc = 1 #num channels in the net I/0
alpha = 1e-5 #learning rate of Lasso
alpha_tv = 1e-1 #TV parameter for net loss

CUDA = torch.cuda.is_available()
print("On GPU: ", CUDA)

if CUDA :
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

filename = "captain"
test_type = "Dropout"
noisy = True
noise_std = 0
if noisy:
    noise_str = "-noisy-" + str(noise_std)
else:
    noise_str = ""

save_loc = "audio_results/" + filename + "-" + test_type + noise_str + "/"

wave_rate, wave_len, wave_res, nc, x0 = inverse_utils.read_wav("audio_data/" + filename + "_8192hz_2s.wav")
if wave_len != 16384 or nc > 1:
    print("ILL-FORMATTED WAV - TRY AGAIN")
    exit(0)

if test_type == 'Dropout' or test_type =='CS':
    num_measurements = [100, 500, 1000, 2000, 4000]
else:
    num_measurements = [wave_len]

x = inverse_utils.normalise(x0, wave_res*8)  #normalise the wave data to [-1,1]

spectrum =np.fft.fft(x[:,0], norm='ortho')
spectrum = abs(spectrum[0:round(len(spectrum)/2)]) # Just first half of the spectrum, as the second is the negative copy

plt.figure()
plt.plot(spectrum, 'r')
plt.xlabel('Frequency (hz)')
plt.title('Original Waveform')
plt.xlim(0, wave_rate/2)
plt.savefig(save_loc + filename + "_freq.jpg")
plt.close()

mse_list = np.zeros((len(num_measurements), 2))

start = time.time()
for i in range(len(num_measurements)):

    phi, A = inverse_utils.get_A(case=test_type, num_measurements=num_measurements[i], original_length=wave_len)

    y = np.dot(phi, x) #create the measurements
    if noisy:
        y = y + inverse_utils.get_noise(num_samples=num_measurements[i], nc=1, std=noise_std)

    x_hat_lasso1 = inverse_utils.run_Lasso(A=A, y=y, output_size=wave_len, alpha=alpha)
    x_hat_lasso2 = inverse_utils.run_Lasso(A=A, y=y, output_size=wave_len, alpha=alpha)
    x_hat_lasso3 = inverse_utils.run_Lasso(A=A, y=y, output_size=wave_len, alpha=alpha)

    mse_lasso1 = np.mean((np.squeeze(x_hat_lasso1) - np.squeeze(x))**2)
    mse_lasso2 = np.mean((np.squeeze(x_hat_lasso2) - np.squeeze(x)) ** 2)
    mse_lasso3 = np.mean((np.squeeze(x_hat_lasso3) - np.squeeze(x)) ** 2)

    mse_lasso = (mse_lasso1 + mse_lasso2 + mse_lasso3)/3.0
    print("\nLasso MSE - " + str(num_measurements[i]) + " :", mse_lasso)

    x_hat_dip1 = dip_utils.run_DIP(phi, y, dtype, NGF = NGF, LR=LR, MOM=MOM, WD=WD, output_size=wave_len, num_measurements=num_measurements[i], CUDA=CUDA, num_iter=NUM_ITER, alpha_tv=alpha_tv)
    x_hat_dip2 = dip_utils.run_DIP(phi, y, dtype, NGF = NGF, LR=LR, MOM=MOM, WD=WD, output_size=wave_len, num_measurements=num_measurements[i], CUDA=CUDA, num_iter=NUM_ITER, alpha_tv=alpha_tv)
    x_hat_dip3 = dip_utils.run_DIP(phi, y, dtype, NGF = NGF, LR=LR, MOM=MOM, WD=WD, output_size=wave_len, num_measurements=num_measurements[i], CUDA=CUDA, num_iter=NUM_ITER, alpha_tv=alpha_tv)

    mse_dip1 = np.mean((np.squeeze(x_hat_dip1) - np.squeeze(x))**2)
    mse_dip2 = np.mean((np.squeeze(x_hat_dip2) - np.squeeze(x)) ** 2)
    mse_dip3 = np.mean((np.squeeze(x_hat_dip3) - np.squeeze(x)) ** 2)

    mse_dip = (mse_dip1 + mse_dip2 + mse_dip3)/3.0
    print("Net MSE - " + str(num_measurements[i]) + " :", mse_dip)

    mse_list[i,0] = mse_lasso
    mse_list[i,1] = mse_dip

end = time.time()
print("Execution Time: ", round(end-start, 2), "seconds")

plt.figure()
plt.plot(num_measurements, mse_list[:, 0], label = "Lasso", color = 'r',  marker='o')
plt.plot(num_measurements, mse_list[:, 1], label = "Net", color = 'b', marker='D')
plt.xlabel("Num Measurements")
plt.ylabel("MSE")
plt.title(filename + "-" + test_type + noise_str  + " Lasso vs. DIP (averaged)")
plt.legend()
plt.savefig(save_loc + filename + "-" + test_type + noise_str + "-" + str(NUM_ITER) + "iter_lasso_net_comp.jpg")
plt.show()
