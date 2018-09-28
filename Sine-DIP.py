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
NUM_ITER = 100 # number iterations
WD = 1e-4 # weight decay for l2-regularization

Z_NUM = 32 # input seed
NGF = 64 # number of filters per layer
ALEX_BATCH_SIZE = 1 # batch size of gradient step
nc = 1 #num channels in the net I/0

#choose the number of samples and periods in the training waveform
WAVE_SIZE = 1024
WAVE_PERIODS = 2


COMP = False

if COMP:
    num_measurements = 64
else:
    num_measurements = WAVE_SIZE


CUDA = torch.cuda.is_available()
print(CUDA)

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


mse = torch.nn.MSELoss().type(dtype)

#num_harmonics selects how many superimposed waveforms to train on, gan selects which net topology to use
def run(num_harmonics = 1, gan = 2):

    mse_log = np.zeros((NUM_ITER)) #track the MSE between the denoised and net output
    best_wave = np.zeros((WAVE_SIZE)) #track the best waveform
    cur_best_mse = 1e6 #track the current best MSE

    if(gan==0):
        net = nets.DCGAN_Straight(Z_NUM, NGF, WAVE_SIZE, nc, num_measurements, cuda = CUDA)
    elif(gan==1):
        net = nets.DCGAN_Funnel(Z_NUM, NGF, WAVE_SIZE, nc, num_measurements, cuda = CUDA)
    elif(gan==2):
        net = nets.DCGAN_Straight_Exhaustive(Z_NUM, NGF, WAVE_SIZE, nc, num_measurements, cuda = CUDA)
    elif(gan==3):
        net = nets.DCGAN_Funnel_Exhaustive(Z_NUM, NGF, WAVE_SIZE, nc, num_measurements, cuda = CUDA)
    else:
        quit()

    net.fc.requires_grad = False

    if CUDA:  # move network to GPU if available
        net.cuda()

    if COMP:
        net.fc.weight.data = (1 / math.sqrt(1.0 * num_measurements)) * torch.randn(num_measurements, WAVE_SIZE * nc)
    else:
        net.fc.weight.data = torch.eye(num_measurements)

    allparams = [x for x in net.parameters()]  # specifies which to compute gradients of
    allparams = allparams[:-1]  # get rid of last item in list (fc layer) because it's memory intensive

    z = Variable(torch.zeros(ALEX_BATCH_SIZE * Z_NUM).type(dtype).view(ALEX_BATCH_SIZE, Z_NUM, 1))
    z.data.normal_().type(dtype)

    # Define optimizer
    optim = torch.optim.RMSprop(allparams, lr=LR, momentum=MOM, weight_decay=WD)

    y0 = get_sinusoid(num_samples=WAVE_SIZE, num_periods=WAVE_PERIODS, num_harmonics = num_harmonics, noisy=True)
    y0_denoised = get_sinusoid(num_samples=WAVE_SIZE, num_periods=WAVE_PERIODS, num_harmonics=num_harmonics, noisy=False)

    MU = get_stats(y0)[0]
    SIGMA = get_stats(y0)[1]

    y = torch.Tensor(y0)
    y = normalise(y, MU, SIGMA)
    y = Variable(y.type(dtype))

    measurements = Variable(torch.mm(y.cpu().data.view(ALEX_BATCH_SIZE, -1), net.fc.weight.data.permute(1, 0)), requires_grad=False)

    if CUDA:  # move measurements to GPU if possible
        measurements = measurements.cuda()

    for i in range(NUM_ITER):
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

print(np.argmin(run()[0]))