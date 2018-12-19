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
#NUM_ITER = 3000 # number iterations
WD = 1 # weight decay for l2-regularization
Z_NUM = 32 # input seed
NGF = 64 # number of filters per layer
nc = 1 #num channels in the net I/0
alpha = 1e-5 #learning rate of Lasso
alpha_tv = 1e-1

CUDA = torch.cuda.is_available()
print("On GPU: ", CUDA)

if CUDA :
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

filename = "speech-testing"
test_type = "Dropout"
noisy = False
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

mse_list = np.zeros((len(num_measurements), 5))

start = time.time()
for i in range(len(num_measurements)):

    phi, A = inverse_utils.get_A(case=test_type, num_measurements=num_measurements[i], original_length=wave_len)

    y = np.dot(phi, x) #create the measurements
    if noisy:
        y = y + inverse_utils.get_noise(num_samples=num_measurements[i], nc=1, std=noise_std)

    x_hat_dip = dip_utils.run_DIP(phi, y, dtype, LR=LR, MOM=MOM, WD=WD, output_size=wave_len, num_measurements=num_measurements[i], CUDA=CUDA, num_iter=1000, alpha_tv=alpha_tv)
    x_hat_dip2 = dip_utils.run_DIP(phi, y, dtype, LR=LR, MOM=MOM, WD=WD, output_size=wave_len,
                                  num_measurements=num_measurements[i], CUDA=CUDA, num_iter=1000, alpha_tv=alpha_tv)
    x_hat_dip3 = dip_utils.run_DIP(phi, y, dtype, LR=LR, MOM=MOM, WD=WD, output_size=wave_len,
                                  num_measurements=num_measurements[i], CUDA=CUDA, num_iter=1000, alpha_tv=alpha_tv)
    mseone1 = np.mean((np.squeeze(x_hat_dip) - np.squeeze(x))**2)
    mseone2 = np.mean((np.squeeze(x_hat_dip2) - np.squeeze(x)) ** 2)
    mseone3 = np.mean((np.squeeze(x_hat_dip3) - np.squeeze(x)) ** 2)
    mseone = (mseone1 + mseone2 + mseone3)/3.0
    print("\n1000 iter - " + str(num_measurements[i]) + " :", mseone)

    x_hat_dip = dip_utils.run_DIP(phi, y, dtype, LR=LR, MOM=MOM, WD=WD, output_size=wave_len, num_measurements=num_measurements[i], CUDA=CUDA, num_iter=2000, alpha_tv=alpha_tv)
    x_hat_dip2 = dip_utils.run_DIP(phi, y, dtype, LR=LR, MOM=MOM, WD=WD, output_size=wave_len,
                                  num_measurements=num_measurements[i], CUDA=CUDA, num_iter=2000, alpha_tv=alpha_tv)
    x_hat_dip3 = dip_utils.run_DIP(phi, y, dtype, LR=LR, MOM=MOM, WD=WD, output_size=wave_len,
                                  num_measurements=num_measurements[i], CUDA=CUDA, num_iter=2000, alpha_tv=alpha_tv)
    msetwo1 = np.mean((np.squeeze(x_hat_dip) - np.squeeze(x))**2)
    msetwo2 = np.mean((np.squeeze(x_hat_dip2) - np.squeeze(x)) ** 2)
    msetwo3 = np.mean((np.squeeze(x_hat_dip3) - np.squeeze(x)) ** 2)
    msetwo = (msetwo1 + msetwo2 + msetwo3)/3.0
    print("2000 iter - " + str(num_measurements[i]) + " :", msetwo)

    x_hat_dip = dip_utils.run_DIP(phi, y, dtype, LR=LR, MOM=MOM, WD=WD, output_size=wave_len, num_measurements=num_measurements[i], CUDA=CUDA, num_iter=3000, alpha_tv=alpha_tv)
    x_hat_dip2 = dip_utils.run_DIP(phi, y, dtype, LR=LR, MOM=MOM, WD=WD, output_size=wave_len,
                                  num_measurements=num_measurements[i], CUDA=CUDA, num_iter=3000, alpha_tv=alpha_tv)
    x_hat_dip3 = dip_utils.run_DIP(phi, y, dtype, LR=LR, MOM=MOM, WD=WD, output_size=wave_len,
                                  num_measurements=num_measurements[i], CUDA=CUDA, num_iter=3000, alpha_tv=alpha_tv)
    msethree1 = np.mean((np.squeeze(x_hat_dip) - np.squeeze(x))**2)
    msethree2 = np.mean((np.squeeze(x_hat_dip2) - np.squeeze(x)) ** 2)
    msethree3 = np.mean((np.squeeze(x_hat_dip3) - np.squeeze(x)) ** 2)
    msethree = (msethree1 + msethree2 + msethree3)/3.0
    print("3000 iter - " + str(num_measurements[i]) + " :", msethree)

    x_hat_dip = dip_utils.run_DIP(phi, y, dtype, LR=LR, MOM=MOM, WD=WD, output_size=wave_len, num_measurements=num_measurements[i], CUDA=CUDA, num_iter=4000, alpha_tv=alpha_tv)
    x_hat_dip2 = dip_utils.run_DIP(phi, y, dtype, LR=LR, MOM=MOM, WD=WD, output_size=wave_len,
                                  num_measurements=num_measurements[i], CUDA=CUDA, num_iter=4000, alpha_tv=alpha_tv)
    x_hat_dip3 = dip_utils.run_DIP(phi, y, dtype, LR=LR, MOM=MOM, WD=WD, output_size=wave_len,
                                  num_measurements=num_measurements[i], CUDA=CUDA, num_iter=4000, alpha_tv=alpha_tv)
    msefour1 = np.mean((np.squeeze(x_hat_dip) - np.squeeze(x))**2)
    msefour2 = np.mean((np.squeeze(x_hat_dip2) - np.squeeze(x)) ** 2)
    msefour3 = np.mean((np.squeeze(x_hat_dip3) - np.squeeze(x)) ** 2)
    msefour = (msefour1 + msefour2 + msefour3)/3.0
    print("4000 iter - " + str(num_measurements[i]) + " :", msefour)

    x_hat_dip = dip_utils.run_DIP(phi, y, dtype, LR=LR, MOM=MOM, WD=WD, output_size=wave_len, num_measurements=num_measurements[i], CUDA=CUDA, num_iter=5000, alpha_tv=alpha_tv)
    x_hat_dip2 = dip_utils.run_DIP(phi, y, dtype, LR=LR, MOM=MOM, WD=WD, output_size=wave_len,
                                  num_measurements=num_measurements[i], CUDA=CUDA, num_iter=5000, alpha_tv=alpha_tv)
    x_hat_dip3 = dip_utils.run_DIP(phi, y, dtype, LR=LR, MOM=MOM, WD=WD, output_size=wave_len,
                                  num_measurements=num_measurements[i], CUDA=CUDA, num_iter=5000, alpha_tv=alpha_tv)
    msefive1 = np.mean((np.squeeze(x_hat_dip) - np.squeeze(x))**2)
    msefive2 = np.mean((np.squeeze(x_hat_dip2) - np.squeeze(x)) ** 2)
    msefive3 = np.mean((np.squeeze(x_hat_dip3) - np.squeeze(x)) ** 2)
    msefive = (msefive1 + msefive2 + msefive3)/3.0
    print("5000 iter - " + str(num_measurements[i]) + " :", msefive)

    mse_list[i, 0] = mseone
    mse_list[i, 1] = msetwo
    mse_list[i, 2] = msethree
    mse_list[i, 3] = msefour
    mse_list[i, 4] = msefive

end = time.time()
print("Execution Time: ", round(end-start, 2), "seconds")

plt.figure()
plt.plot(num_measurements, mse_list[:, 0], label = "1000", color = 'r', marker = 'o')
plt.plot(num_measurements, mse_list[:, 1], label = "2000", color = 'b', marker = 'D')
plt.plot(num_measurements, mse_list[:, 2], label = "3000", color = 'g', marker = '+')
plt.plot(num_measurements, mse_list[:, 3], label = "4000", color = 'k', marker = '^')
plt.plot(num_measurements, mse_list[:, 4], label = "5000", color = 'y', marker = 's')
plt.xlabel("Num Measurements")
plt.ylabel("MSE")
plt.title(filename + "-" + test_type + noise_str  + " Iter parameter comparison")
plt.legend()
plt.savefig(save_loc + filename + "-" + test_type + noise_str + "- iter_comp.jpg")
plt.show()






