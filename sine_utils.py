import random

import torch

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import math


#Starts with len 16, straight 64 channels, length doubles each time
class DCGAN_Straight(nn.Module):
    def __init__(self, nz, ngf=64, output_size=1024, nc=1, num_measurements=64, cuda = True):
        super(DCGAN_Straight, self).__init__()
        self.nc = nc
        self.output_size = output_size
        self.CUDA = cuda

        # Deconv Layers: (in_channels, out_channels, kernel_size, stride, padding, bias = false)
        # Inputs: R^(N x Cin x Lin), Outputs: R^(N, Cout, Lout) s.t. Lout = (Lin - 1)*stride - 2*padding + kernel_size

        self.conv1 = nn.ConvTranspose1d(nz, ngf, 16, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm1d(ngf)
        # LAYER 1: input: (random) zϵR^(nzx1), output: x1ϵR^(64x16) (channels x length)

        self.conv2 = nn.ConvTranspose1d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn2 = nn.BatchNorm1d(ngf)
        # LAYER 2: input: x1ϵR^(64x16), output: x2ϵR^(64x32) (channels x length)

        self.conv3 = nn.ConvTranspose1d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn3 = nn.BatchNorm1d(ngf)
        # LAYER 3: input: x2ϵR^(64x32), output: x3ϵR^(64x64) (channels x length)

        self.conv4 = nn.ConvTranspose1d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn4 = nn.BatchNorm1d(ngf)
        # LAYER 4: input: x3ϵR^(64x64), output: x4ϵR^(64x128) (channels x length)

        self.conv5 = nn.ConvTranspose1d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn5 = nn.BatchNorm1d(ngf)
        # LAYER 5: input: x4ϵR^(64x128), output: x5ϵR^(64x256) (channels x length)

        self.conv6 = nn.ConvTranspose1d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn6 = nn.BatchNorm1d(ngf)
        # LAYER 6: input: x5ϵR^(64x256), output: x6ϵR^(64x512) (channels x length)

        self.conv7 = nn.ConvTranspose1d(ngf, nc, 4, 2, 1, bias=False)  # output is image
        # LAYER 7: input: x6ϵR^(64x512), output: (sinusoid) G(z,w)ϵR^(1x1024) (channels x length)

        self.fc = nn.Linear(output_size * nc, num_measurements, bias=False)  # output is A; measurement matrix
        # each entry should be drawn from a Gaussian (random noisy measurements)
        # don't compute gradient of self.fc! memory issues

    def forward(self, x):
        input_size = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.tanh(self.conv7(x))

        return x

    def measurements(self, x):
        # this gives the image - make it a single row vector of appropriate length
        y = self.forward(x).view(1, -1)
        y = y.cpu()

        # pass thru FC layer - returns A*image
        meas = self.fc(y)

        if self.CUDA:
            return meas.cuda()
        else:
            return meas


#starts with len 16, 2048 channels shrinking, len doubles each time
class DCGAN_Funnel(nn.Module):
    def __init__(self, nz, ngf=64, output_size=1024, nc=1, num_measurements=64, cuda = True):
        super(DCGAN_Funnel, self).__init__()
        self.nc = nc
        self.output_size = output_size
        self.CUDA = cuda

        # Deconv Layers: (in_channels, out_channels, kernel_size, stride, padding, bias = false)
        # Inputs: R^(N x Cin x Lin), Outputs: R^(N, Cout, Lout) s.t. Lout = (Lin - 1)*stride - 2*padding + kernel_size

        self.conv1 = nn.ConvTranspose1d(nz, ngf*32, 16, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm1d(ngf*32)
        # LAYER 1: input: (random) zϵR^(nzx1), output: x1ϵR^(2048x16) (channels x length)

        self.conv2 = nn.ConvTranspose1d(ngf*32, ngf*16, 6, 2, 2, bias=False)
        self.bn2 = nn.BatchNorm1d(ngf*16)
        # LAYER 2: input: x1ϵR^(2048x16), output: x2ϵR^(1024x32) (channels x length)

        self.conv3 = nn.ConvTranspose1d(ngf*16, ngf*8, 6, 2, 2, bias=False)
        self.bn3 = nn.BatchNorm1d(ngf*8)
        # LAYER 3: input: x2ϵR^(1024x32), output: x3ϵR^(512x64) (channels x length)

        self.conv4 = nn.ConvTranspose1d(ngf*8, ngf*4, 6, 2, 2, bias=False)
        self.bn4 = nn.BatchNorm1d(ngf*4)
        # LAYER 4: input: x3ϵR^(512x64), output: x4ϵR^(256x128) (channels x length)

        self.conv5 = nn.ConvTranspose1d(ngf*4, ngf*2, 6, 2, 2, bias=False)
        self.bn5 = nn.BatchNorm1d(ngf*2)
        # LAYER 5: input: x4ϵR^(256x128), output: x5ϵR^(128x256) (channels x length)

        self.conv6 = nn.ConvTranspose1d(ngf*2, ngf, 6, 2, 2, bias=False)
        self.bn6 = nn.BatchNorm1d(ngf)
        # LAYER 6: input: x5ϵR^(128x256), output: x6ϵR^(64x512) (channels x length)

        self.conv7 = nn.ConvTranspose1d(ngf, nc, 4, 2, 1, bias=False)  # output is image
        # LAYER 7: input: x6ϵR^(64x512), output: (sinusoid) G(z,w)ϵR^(1x1024) (channels x length)

        self.fc = nn.Linear(output_size * nc, num_measurements, bias=False)  # output is A; measurement matrix
        # each entry should be drawn from a Gaussian (random noisy measurements)
        # don't compute gradient of self.fc! memory issues

    def forward(self, x):
        input_size = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.tanh(self.conv7(x))

        return x

    def measurements(self, x):
        # this gives the image - make it a single row vector of appropriate length
        y = self.forward(x).view(1, -1)
        y = y.cpu()

        # pass thru FC layer - returns A*image
        meas = self.fc(y)

        if self.CUDA:
            return meas.cuda()
        else:
            return meas


#starts with len 4, 64 straight channels, len doubles each time
class DCGAN_Straight_Exhaustive(nn.Module):
    def __init__(self, nz, ngf=64, output_size=1024, nc=1, num_measurements=64, cuda = True):
        super(DCGAN_Straight_Exhaustive, self).__init__()
        self.nc = nc
        self.output_size = output_size
        self.CUDA = cuda

        # Deconv Layers: (in_channels, out_channels, kernel_size, stride, padding, bias = false)
        # Inputs: R^(N x Cin x Lin), Outputs: R^(N, Cout, Lout) s.t. Lout = (Lin - 1)*stride - 2*padding + kernel_size

        self.conv1 = nn.ConvTranspose1d(nz, ngf, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm1d(ngf)
        # LAYER 1: input: (random) zϵR^(nzx1), output: x1ϵR^(64x4) (channels x length)

        self.conv2 = nn.ConvTranspose1d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn2 = nn.BatchNorm1d(ngf)
        # LAYER 2: input: x1ϵR^(64x4), output: x2ϵR^(64x8) (channels x length)

        self.conv3 = nn.ConvTranspose1d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn3 = nn.BatchNorm1d(ngf)
        # LAYER 3: input: x1ϵR^(64x8), output: x2ϵR^(64x16) (channels x length)

        self.conv4 = nn.ConvTranspose1d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn4 = nn.BatchNorm1d(ngf)
        # LAYER 4: input: x1ϵR^(64x16), output: x2ϵR^(64x32) (channels x length)

        self.conv5 = nn.ConvTranspose1d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn5 = nn.BatchNorm1d(ngf)
        # LAYER 5: input: x2ϵR^(64x32), output: x3ϵR^(64x64) (channels x length)

        self.conv6 = nn.ConvTranspose1d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn6 = nn.BatchNorm1d(ngf)
        # LAYER 6: input: x3ϵR^(64x64), output: x4ϵR^(64x128) (channels x length)

        self.conv7 = nn.ConvTranspose1d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn7 = nn.BatchNorm1d(ngf)
        # LAYER 7: input: x4ϵR^(64x128), output: x5ϵR^(64x256) (channels x length)

        self.conv8 = nn.ConvTranspose1d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn8 = nn.BatchNorm1d(ngf)
        # LAYER 8: input: x5ϵR^(64x256), output: x6ϵR^(64x512) (channels x length)

        self.conv9 = nn.ConvTranspose1d(ngf, nc, 4, 2, 1, bias=False)  # output is image
        # LAYER 9: input: x6ϵR^(64x512), output: (sinusoid) G(z,w)ϵR^(1x1024) (channels x length)

        self.fc = nn.Linear(output_size * nc, num_measurements, bias=False)  # output is A; measurement matrix
        # each entry should be drawn from a Gaussian (random noisy measurements)
        # don't compute gradient of self.fc! memory issues

    def forward(self, x):
        input_size = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.tanh(self.conv9(x))

        return x

    def measurements(self, x):
        # this gives the image - make it a single row vector of appropriate length
        y = self.forward(x).view(1, -1)
        y = y.cpu()

        # pass thru FC layer - returns A*image
        meas = self.fc(y)

        if self.CUDA:
            return meas.cuda()
        else:
            return meas


#starts with len 4, 2048 channels shrinking, len doubles each time
class DCGAN_Funnel_Exhaustive(nn.Module):
    def __init__(self, nz, ngf=64, output_size=1024, nc=1, num_measurements=64, cuda = True):
        super(DCGAN_Funnel_Exhaustive, self).__init__()
        self.nc = nc
        self.output_size = output_size
        self.CUDA = cuda

        # Deconv Layers: (in_channels, out_channels, kernel_size, stride, padding, bias = false)
        # Inputs: R^(N x Cin x Lin), Outputs: R^(N, Cout, Lout) s.t. Lout = (Lin - 1)*stride - 2*padding + kernel_size

        self.conv1 = nn.ConvTranspose1d(nz, ngf*32, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm1d(ngf*32)
        # LAYER 1: input: (random) zϵR^(nzx1), output: x1ϵR^(64x4) (channels x length)

        self.conv2 = nn.ConvTranspose1d(ngf*32, ngf*16, 6, 2, 2, bias=False)
        self.bn2 = nn.BatchNorm1d(ngf*16)
        # LAYER 2: input: x1ϵR^(64x4), output: x2ϵR^(64x8) (channels x length)

        self.conv3 = nn.ConvTranspose1d(ngf*16, ngf*8, 6, 2, 2, bias=False)
        self.bn3 = nn.BatchNorm1d(ngf*8)
        # LAYER 3: input: x1ϵR^(64x8), output: x2ϵR^(64x16) (channels x length)

        self.conv4 = nn.ConvTranspose1d(ngf*8, ngf*4, 6, 2, 2, bias=False)
        self.bn4 = nn.BatchNorm1d(ngf*4)
        # LAYER 4: input: x1ϵR^(64x16), output: x2ϵR^(64x32) (channels x length)

        self.conv5 = nn.ConvTranspose1d(ngf*4, ngf*2, 6, 2, 2, bias=False)
        self.bn5 = nn.BatchNorm1d(ngf*2)
        # LAYER 5: input: x2ϵR^(64x32), output: x3ϵR^(64x64) (channels x length)

        self.conv6 = nn.ConvTranspose1d(ngf*2, ngf, 6, 2, 2, bias=False)
        self.bn6 = nn.BatchNorm1d(ngf)
        # LAYER 6: input: x3ϵR^(64x64), output: x4ϵR^(64x128) (channels x length)

        self.conv7 = nn.ConvTranspose1d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn7 = nn.BatchNorm1d(ngf)
        # LAYER 7: input: x4ϵR^(64x128), output: x5ϵR^(64x256) (channels x length)

        self.conv8 = nn.ConvTranspose1d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn8 = nn.BatchNorm1d(ngf)
        # LAYER 8: input: x5ϵR^(64x256), output: x6ϵR^(64x512) (channels x length)

        self.conv9 = nn.ConvTranspose1d(ngf, nc, 4, 2, 1, bias=False)  # output is image
        # LAYER 9: input: x6ϵR^(64x512), output: (sinusoid) G(z,w)ϵR^(1x1024) (channels x length)

        self.fc = nn.Linear(output_size * nc, num_measurements, bias=False)  # output is A; measurement matrix
        # each entry should be drawn from a Gaussian (random noisy measurements)
        # don't compute gradient of self.fc! memory issues

    def forward(self, x):
        input_size = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.tanh(self.conv9(x))

        return x

    def measurements(self, x):
        # this gives the image - make it a single row vector of appropriate length
        y = self.forward(x).view(1, -1)
        y = y.cpu()

        # pass thru FC layer - returns A*image
        meas = self.fc(y)

        if self.CUDA:
            return meas.cuda()
        else:
            return meas


#num_harmonics selects how many superimposed waveforms to train on, gan selects which net topology to use
def run(num_harmonics = 1, gan = 2, num_iter = 80, wave_size = 1024, wave_periods = 2, std = 0.1, num_measurements = 1024, cuda = True, compressed = False, compressed_noisy = False, plot_waves = False, filters_per_layer = 64, batch_size = 1, num_channels = 1, seed_size = 32, learning_rate = 1e-3, momentum = 0.9, weight_decay = 1e-4):

    mse_log = np.zeros((num_iter)) #track the MSE between the denoised and net output
    best_wave = np.zeros((wave_size)) #track the best waveform
    cur_best_mse = 1e6 #track the current best MSE

    #select net topology
    if(gan==0):
        net = DCGAN_Straight(seed_size, filters_per_layer, wave_size, num_channels, num_measurements, cuda = cuda)
    elif(gan==1):
        net = DCGAN_Funnel(seed_size, filters_per_layer, wave_size, num_channels, num_measurements, cuda = cuda)
    elif(gan==2):
        net = DCGAN_Straight_Exhaustive(seed_size, filters_per_layer, wave_size, num_channels, num_measurements, cuda = cuda)
    elif(gan==3):
        net = DCGAN_Funnel_Exhaustive(seed_size, filters_per_layer, wave_size, num_channels, num_measurements, cuda = cuda)
    else:
        quit()

    net.fc.requires_grad = False

    if cuda:  # move network to GPU if available
        net.cuda()

    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor


    #initialize the fc layer as the measurement matrix A
    if compressed:
        if compressed_noisy:
            net.fc.weight.data = (1 / math.sqrt(1.0 * num_measurements)) * torch.randn(num_measurements, wave_size * num_channels) #measurement matrix is normalized gaussian R^(num_measurements, wave_size*num_channels)
        else:
            kept_samples = random.sample(range(0, wave_size), num_measurements) #randomly select num_measurements samples to keep
            net.fc.weight.data = torch.eye(wave_size)[kept_samples,:] #grab rows corresponding to index of randomly kept samples from identity
    else:
        net.fc.weight.data = torch.eye(wave_size)

    allparams = [x for x in net.parameters()]  # specifies which to compute gradients of
    allparams = allparams[:-1]  # get rid of last item in list (fc layer) because it's memory intensive

    z = Variable(torch.zeros(batch_size * seed_size).type(dtype).view(batch_size, seed_size, 1))
    z.data.normal_().type(dtype)

    # Define optimizer
    optim = torch.optim.RMSprop(allparams, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    y0 = get_sinusoid(num_samples=wave_size, num_periods=wave_periods, num_harmonics = num_harmonics, noisy=(not compressed), std = std)
    y0_denoised = get_sinusoid(num_samples=wave_size, num_periods=wave_periods, num_harmonics=num_harmonics, noisy=False, std = std)

    # Plot both noisy (blue) and denoised (red) waveforms
    if plot_waves:
        plt.plot(np.arange(1024), y0)
        plt.plot(np.arange(1024), y0_denoised, color='r')
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.show()

    MU = get_stats(y0)[0]
    SIGMA = get_stats(y0)[1]

    #normalize the sinusoid to [-1,1]
    y = torch.Tensor(y0)
    y = normalise(y, MU, SIGMA)
    y = Variable(y.type(dtype))

    #get the measurements y (A*y in the compressed case) to compare the net output with
    measurements = Variable(torch.mm(y.cpu().data.view(batch_size, -1), net.fc.weight.data.permute(1, 0)), requires_grad=False)

    if cuda:
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


def get_sinusoid(num_samples, num_periods, num_harmonics = 1, noisy = True, std = 0.1, mean = 0):
    Fs = num_samples
    x = np.arange(num_samples)

    y = np.zeros((num_samples))

    for i in range(num_harmonics):
        y += np.sin(2 * np.pi * (2**i) * num_periods * x / Fs)

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