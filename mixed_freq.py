import numpy as np
import matplotlib.pyplot as plt
import sine_utils as nets
import torch
import math

LR = 1e-3 # learning rate
MOM = 0.9 # momentum
NUM_ITER = 150 # number iterations
WD = 1e-4 # weight decay for l2-regularization

Z_NUM = 32 # input seed
NGF = 64 # number of filters per layer
ALEX_BATCH_SIZE = 1 # batch size of gradient step
nc = 1 #num channels in the net I/0

#choose the number of samples and periods in the training waveform
WAVE_SIZE = 1024
WAVE_PERIODS = 2

harmonics_list = range(1, 10) #list of number of harmonics to try

mse_list = np.zeros((len(harmonics_list), NUM_ITER)) #store the mse log of each harmonic signal

CUDA = torch.cuda.is_available()
print("Using CUDA: ", CUDA)

for i in range(len(harmonics_list)):
    mse_list[i, :] = nets.run(cuda = CUDA, num_harmonics=harmonics_list[i], num_iter=NUM_ITER, wave_periods=WAVE_PERIODS)[0]

x_axis = np.arange(NUM_ITER)
for i in range(len(harmonics_list)):
    plt.plot(x_axis, mse_list[i, :], label = str(i+1), color = str((1.0*i)/len(harmonics_list)))

plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.legend()
plt.ylim(0,5e-2)
plt.show()