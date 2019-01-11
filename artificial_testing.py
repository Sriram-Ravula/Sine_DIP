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
import collections
from scipy.signal import chirp

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
wave_len = 16384

CUDA = torch.cuda.is_available()
print("On GPU: ", CUDA)

if CUDA :
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

filename = "Mackey-Glass"
test_type = "Imputation"
noisy = False
noise_std = 0
if noisy:
    noise_str = "-noisy-" + str(noise_std)
else:
    noise_str = ""

save_loc = "audio_results/" + filename + "-" + test_type + noise_str + "/"

t = np.linspace(0, 2, 16384)
x0 = chirp(t, f0=750, f1=250, t1=2, method='linear')
x = np.zeros((wave_len, 1))
x[:,0] = x0

if test_type == 'Imputation' or test_type =='CS':
    num_measurements = [100, 500, 1000, 2000, 4000]
else:
    num_measurements = [wave_len]

#x = inverse_utils.normalise(x0, wave_res*8)  #normalise the wave data to [-1,1]
#x = inverse_utils.normalise(x0)

spectrum =np.fft.fft(x[:,0], norm='ortho')
spectrum = abs(spectrum[0:round(len(spectrum)/2)]) # Just first half of the spectrum, as the second is the negative copy

plt.figure()
plt.plot(spectrum, 'r')
plt.xlabel('Frequency (hz)')
plt.title('Original Waveform')
plt.xlim(0, 8192)
#plt.savefig(save_loc + filename + "_freq.jpg")
plt.close()

mse_list = np.zeros((len(num_measurements), 2))

start = time.time()
for i in range(len(num_measurements)):

    if test_type != "Imputation":
        phi, A = inverse_utils.get_A(case=test_type, num_measurements=num_measurements[i], original_length=wave_len)
    else:
        phi, A, kept_samples = inverse_utils.get_A(case=test_type, num_measurements=num_measurements[i], original_length=wave_len)

    y = np.dot(phi, x) #create the measurements

    num_instances = 5  # how many instances of lasso and DIP we wish to run to average results
    mse_lasso = 0
    mse_DIP = 0
    for t in range(num_instances):

        x_hat_lasso = inverse_utils.run_Lasso(A=A, y=y, output_size=wave_len, alpha=alpha)

        if test_type == "Imputation":  # for imputation, we want to add back in the values we know to the predictions
            removed_samples = [x for x in range(0, wave_len) if x not in kept_samples]
            mse_lasso = mse_lasso + np.mean((np.squeeze(x_hat_lasso)[removed_samples] - np.squeeze(x)[removed_samples]) ** 2)

        else:
            mse_lasso = mse_lasso + np.mean((np.squeeze(x_hat_lasso) - np.squeeze(x)) ** 2)

        x_hat_DIP = dip_utils.run_DIP(phi, y, dtype, NGF=NGF, LR=LR, MOM=MOM, WD=WD, output_size=wave_len, num_measurements=num_measurements[i], CUDA=CUDA, num_iter=NUM_ITER, alpha_tv=alpha_tv)

        if test_type == "Imputation":  # for imputation, we want to add back in the values we know to the predictions
            mse_DIP = mse_DIP + np.mean((np.squeeze(x_hat_DIP)[removed_samples] - np.squeeze(x)[removed_samples]) ** 2)

        else:
            mse_DIP = mse_DIP + np.mean((np.squeeze(x_hat_DIP) - np.squeeze(x)) ** 2)

    mse_lasso = mse_lasso / float(num_instances)
    mse_DIP = mse_DIP / float(num_instances)

    mse_list[i,0] = mse_lasso
    mse_list[i,1] = mse_DIP

    print("\nNet MSE - " + str(num_measurements[i]) + " :", mse_DIP)
    print("Lasso MSE - " + str(num_measurements[i]) + " :", mse_lasso)

end = time.time()
print("Execution Time: ", round(end-start, 2), "seconds")

plt.figure()
plt.plot(num_measurements, mse_list[:, 0], label = "Lasso", color = 'r',  marker='o')
plt.plot(num_measurements, mse_list[:, 1], label = "Net", color = 'b', marker='D')
#plt.errorbar(num_measurements, mse_list[:, 0], yerr=error_lasso, fmt='ro-', label="Lasso")
#plt.errorbar(num_measurements, mse_list[:, 1], yerr=error_dip, fmt='bD-', label="Net")
plt.xlabel("Num Measurements")
plt.ylabel("MSE")
plt.title(filename + "-" + test_type + noise_str  + " Lasso vs. DIP (averaged)")
plt.legend()
#plt.savefig(save_loc + filename + "-" + test_type + noise_str + "-" + str(NUM_ITER) + "iter_lasso_net_comp.jpg")
plt.show()
