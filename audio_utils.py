import random

import torch

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import math

import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg

import wavio

from sklearn.linear_model import Lasso


class DCGAN_Audio_Straight(nn.Module):
    def __init__(self, nz=32, ngf=64, output_size=16384, nc=1, num_measurements=1000, cuda = True):
        super(DCGAN_Audio_Straight, self).__init__()
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

        self.conv9 = nn.ConvTranspose1d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn9 = nn.BatchNorm1d(ngf)
        # LAYER 9: input: x5ϵR^(64x512), output: x6ϵR^(64x1024) (channels x length)

        self.conv10 = nn.ConvTranspose1d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn10 = nn.BatchNorm1d(ngf)
        # LAYER 10: input: x5ϵR^(64x1024), output: x6ϵR^(64x2048) (channels x length)

        self.conv11 = nn.ConvTranspose1d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn11 = nn.BatchNorm1d(ngf)
        # LAYER 11: input: x5ϵR^(64x2048), output: x6ϵR^(64x4096) (channels x length)

        self.conv12 = nn.ConvTranspose1d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn12 = nn.BatchNorm1d(ngf)
        # LAYER 12: input: x5ϵR^(64x4096), output: x6ϵR^(64x8192) (channels x length)

        self.conv13 = nn.ConvTranspose1d(ngf, nc, 4, 2, 1, bias=False)  # output is image
        # LAYER 13: input: x6ϵR^(64x8192), output: (sinusoid) G(z,w)ϵR^(ncx16384) (channels x length)

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
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = F.tanh(self.conv13(x))

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

def read_wav(filename):
    wave = wavio.read(filename)
    rate = wave.rate
    length = wave.data.shape[0]
    resolution = wave.sampwidth
    nc = wave.data.shape[1]
    y0 = wave.data

    return [rate, length, resolution, nc, y0]

def run_Lasso(A, measurements, wave_res = 2, wave_rate = 8192, num_measurements = 1000, output_size = 16834, num_channels = 1, alpha = 0.001):
    lasso = Lasso(alpha=alpha)
    lasso.fit(A, measurements)

    x_hat = np.array(lasso.coef_).reshape(output_size * num_channels)
    x_hat = spfft.idct(x_hat, norm='ortho', axis=0)
    x_hat = x_hat.reshape(-1, 1)

    #wavio.write("Audio_recon/Lasso_" + str(num_measurements) + "_testR.wav", x_hat, wave_rate, sampwidth=wave_res)

    spectrum = np.fft.fft(x_hat[:, 0])
    spectrum = abs(spectrum[0:round(len(spectrum) / 2)])  # Just first half of the spectrum, as the second is the negative copy

    plt.figure()
    plt.plot(spectrum, 'b')
    plt.xlabel('Frequency (hz)')
    plt.title("Lasso - " + str(num_measurements) + " measurements")
    plt.xlim(0, wave_rate / 2)
    plt.savefig("Freq_recon/bass-dropout2-3000iter/Lasso_" + str(num_measurements) + "_bass-dropout.jpg")
    plt.close()

    return x_hat

def run_DIP(A, y, y0, dtype, LR = 5e-4, MOM = 0.9, WD = 1e-4, num_channels = 1, wave_len = 16384, num_measurements = 1000, wave_rate = 8192, wave_res = 2, CUDA = True, num_iter = 5000):
    net = DCGAN_Audio_Straight(output_size = wave_len, num_measurements = num_measurements)
    net.fc.requires_grad = False

    if CUDA:
        net.cuda()

    net.fc.weight.data = A

    allparams = [x for x in net.parameters()]  # specifies which to compute gradients of
    allparams = allparams[:-1]  # get rid of last item in list (fc layer) because it's memory intensive

    # Define input seed z as Torch variable, normalize
    z = Variable(torch.zeros(32).type(dtype).view(1, 32, 1))
    z.data.normal_().type(dtype)

    # Define optimizer
    optim = torch.optim.RMSprop(allparams, lr=LR, momentum=MOM, weight_decay=WD)

    measurements = Variable(torch.mm(y.cpu().data.view(1, -1), net.fc.weight.data.permute(1, 0)), requires_grad=False)

    if CUDA:  # move measurements to GPU if possible
        measurements = measurements.cuda()

    mse_log = np.zeros((num_iter))

    MU, SIGMA = get_stats(y0)

    mse = torch.nn.MSELoss().type(dtype)

    for i in range(num_iter):
        optim.zero_grad()  # clears graidents of all optimized variables
        out = net(z)  # produces wave (in form of data tensor) i.e. G(z,w)

        loss = mse(net.measurements(z), measurements)  # calculate loss between AG(z,w) and Ay

        # DCGAN output is in [-1,1]. Renormalise to [0,1] before plotting
        wave = out[0].detach().reshape(-1, num_channels).cpu()
        wave = renormalise(wave, MU, SIGMA)

        mse_log[i] = np.mean(
            (normalise(y0, MU, SIGMA).reshape((1, -1)) - normalise(wave, MU, SIGMA).reshape((1, -1))) ** 2)

        if (i == num_iter - 1):
            spectrum = np.fft.fft(wave[:, 0])
            spectrum = abs(spectrum[0:round(len(spectrum) / 2)])  # Just first half of the spectrum, as the second is the negative copy

            plt.figure()
            plt.plot(spectrum, 'b')
            plt.xlabel('Frequency (hz)')
            plt.title("Net - " + str(num_measurements) + " measurements")
            plt.xlim(0, wave_rate / 2)
            plt.savefig("Freq_recon/bass-dropout2-3000iter/Net_" + str(num_measurements) + "_bass-dropout.jpg")
            plt.close()

            #wavio.write("Audio_recon/Net_" + str(num_measurements) + "_testR.wav", wave, wave_rate, sampwidth = wave_res)

        loss.backward()
        optim.step()

    return(mse_log)

#returns measurement matrices for both convnet and Lasso cases
def get_A(compressed = True, noisy = False, num_measurements = 1000, original_length = 16384, num_channels = 1):

    if compressed:
        if noisy:
            torch_meas = (1 / math.sqrt(1.0 * num_measurements)) * torch.randn(num_measurements, original_length* num_channels)

            samp_matrix = torch_meas.numpy()

            cpu_meas = np.matmul(samp_matrix, spfft.idct(np.identity(original_length*num_channels), norm = 'ortho', axis=0))

            return [torch_meas, cpu_meas]
        else:
            kept_samples = random.sample(range(0, original_length), num_measurements)

            lasso_meas = spfft.idct(np.identity(original_length * num_channels), norm='ortho', axis=0)[kept_samples, :]  # grab rows corresponding to index of randomly

            torch_meas = torch.eye(original_length * num_channels)[kept_samples,:]

            return [torch_meas, lasso_meas]
    else:
        torch_meas = torch.eye(original_length * num_channels)
        lasso_meas = spfft.idct(np.identity(original_length*num_channels), norm='ortho', axis=0)

        return [torch_meas, lasso_meas]

def get_stats(x):
    chans = x.shape[1]

    a = np.zeros((chans))
    b = np.zeros((chans))
    mu = np.zeros((chans))
    sigma = np.zeros((chans))

    for c in range(chans):
        a[c] = np.min(x[:, c])
        b[c] = np.max(x[:, c])
        mu[c] = (a[c] + b[c]) / 2.0
        sigma[c] = (b[c] - a[c]) / 2.0

    return [mu, sigma]

def normalise(x, mean, std):
    normalised = np.zeros((x.shape))
    chans = x.shape[1]

    for c in range(chans):
        normalised[:, c] = (x[:, c] - mean[c]) / std[c]

    return normalised

def renormalise(x, mean, std):
    normalised = np.zeros((x.shape))
    chans = x.data.shape[1]

    for c in range(chans):
        normalised[:, c] = x[:, c] * std[c] + mean[c]

    return normalised