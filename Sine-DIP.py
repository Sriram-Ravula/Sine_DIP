import matplotlib.pyplot as plt
import numpy as np
import random
import time
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

start = time.time()
best_log = np.zeros((50))
for i in range(50):
    best_log[i] = np.argmin(nets.run()[0])
end = time.time()

print("time elapsed: ", start-end)
print("Best Iteration: ", best_log)
print("Mean Best Iteration: ", np.mean(best_log))
#print("best iteration: ", np.argmin(run()[0]))