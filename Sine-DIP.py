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

#import torch_dip_utils as utils
import utils
import sine_utils as nets
import math


#set up hyperparameters, net input/output sizes, and whether the problem is compressed sensing

LR = 1e-3 # learning rate
MOM = 0.9 # momentum
NUM_ITER = 80 # number iterations
WD = 1e-4 # weight decay for l2-regularization

Z_NUM = 32 # input seed
NGF = 64 # number of filters per layer
ALEX_BATCH_SIZE = 1 # batch size of gradient step
nc = 1 #num channels in the net I/0

#choose the number of samples and periods in the training waveform
WAVE_SIZE = 1024
WAVE_PERIODS = 1


COMP = False

if COMP:
    NUM_MEASUREMENTS = 64
else:
    NUM_MEASUREMENTS = WAVE_SIZE


CUDA = torch.cuda.is_available()
print("Using CUDA: ", CUDA)

#save the correct datatype depending on CPU or GPU execution
if CUDA :
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor


def get_sinusoid(num_samples, num_periods, num_harmonics = 1, noisy = True, std = 0.1, mean = 0):
    Fs = num_samples
    x = np.arange(num_samples)

    y = np.zeros((num_samples))

    for i in range(num_harmonics):
        y += np.sin(2 * np.pi * (i+1) * num_periods * x / Fs)

    if noisy:
        y += (std * np.random.randn(num_samples)) + mean

    return y


def get_stats(x):
    a = np.min(x)
    b = np.max(x)
    mu = (a+b)/2.0
    sigma = (b-a)/2.0
    return [mu, sigma]

def normalise(x, mu, sigma):
    return (x-mu)/sigma

def renormalise(x, mu, sigma):
    return x*sigma + mu

#num_harmonics selects how many superimposed waveforms to train on, gan selects which net topology to use
def run(num_harmonics = 1, gan = 2, num_iter = 80, wave_size = 1024, wave_periods = 2, num_measurements = 1024, cuda = True, compressed = False, plot_waves = False, filters_per_layer = 64, batch_size = 1, num_channels = 1, seed_size = 32):

    mse_log = np.zeros((num_iter)) #track the MSE between the denoised and net output
    best_wave = np.zeros((wave_size)) #track the best waveform
    cur_best_mse = 1e6 #track the current best MSE

    if(gan==0):
        net = nets.DCGAN_Straight(seed_size, filters_per_layer, wave_size, num_channels, num_measurements, cuda = cuda)
    elif(gan==1):
        net = nets.DCGAN_Funnel(seed_size, filters_per_layer, wave_size, num_channels, num_measurements, cuda = cuda)
    elif(gan==2):
        net = nets.DCGAN_Straight_Exhaustive(seed_size, filters_per_layer, wave_size, num_channels, num_measurements, cuda = cuda)
    elif(gan==3):
        net = nets.DCGAN_Funnel_Exhaustive(seed_size, filters_per_layer, wave_size, num_channels, num_measurements, cuda = cuda)
    else:
        quit()

    net.fc.requires_grad = False

    if cuda:  # move network to GPU if available
        net.cuda()

    if compressed:
        net.fc.weight.data = (1 / math.sqrt(1.0 * num_measurements)) * torch.randn(num_measurements, wave_size * num_channels)
    else:
        net.fc.weight.data = torch.eye(wave_size)

    allparams = [x for x in net.parameters()]  # specifies which to compute gradients of
    allparams = allparams[:-1]  # get rid of last item in list (fc layer) because it's memory intensive

    z = Variable(torch.zeros(batch_size * seed_size).type(dtype).view(batch_size, seed_size, 1))
    z.data.normal_().type(dtype)

    # Define optimizer
    optim = torch.optim.RMSprop(allparams, lr=LR, momentum=MOM, weight_decay=WD)

    y0 = get_sinusoid(num_samples=wave_size, num_periods=wave_periods, num_harmonics = num_harmonics, noisy=True)
    y0_denoised = get_sinusoid(num_samples=wave_size, num_periods=wave_periods, num_harmonics=num_harmonics, noisy=False)

    # Plot both noisy (blue) and denoised (red) waveforms
    if plot_waves:
        plt.plot(np.arange(1024), y0)
        plt.plot(np.arange(1024), y0_denoised, color='r')
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.show()

    MU = get_stats(y0)[0]
    SIGMA = get_stats(y0)[1]

    y = torch.Tensor(y0)
    y = normalise(y, MU, SIGMA)
    y = Variable(y.type(dtype))

    measurements = Variable(torch.mm(y.cpu().data.view(batch_size, -1), net.fc.weight.data.permute(1, 0)), requires_grad=False)

    if cuda:  # move measurements to GPU if possible
        measurements = measurements.cuda()

    mse = torch.nn.MSELoss().type(dtype)

    for i in range(num_iter):
        optim.zero_grad()  # clears graidents of all optimized variables
        out = net(z)  # produces wave (in form of data tensor) i.e. G(z,w)

        loss = mse(net.measurements(z), measurements)  # calculate loss between AG(z,w) and Ay

        # DCGAN output is in [-1,1]. Renormalise to [0,1] before plotting
        wave = renormalise(out, MU, SIGMA).data[0].cpu().numpy()[0, :]

        cur_mse = np.mean((y0_denoised - wave) ** 2)

        mse_log[i] = cur_mse

        if (cur_mse <= cur_best_mse):
            best_wave = wave
            cur_best_mse = cur_mse

        loss.backward()
        optim.step()

    return [mse_log, best_wave]

best_log = np.zeros((50))
for i in range(50):
    best_log[i] = np.argmin(run()[0])

print("Best Iteration: ", best_log)
print("Mean Best Iteration: ", np.mean(best_log))
#print("best iteration: ", np.argmin(run()[0]))