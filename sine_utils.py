import numpy as np
import os
import errno
import parser

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets,transforms

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