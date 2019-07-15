import matplotlib.pyplot as plt
import numpy as np
import Gen_sonar as s

import torch
import inverse_utils
import dip_utils
import time
import pywt
import scipy.fftpack as spfft
from scipy.interpolate import interp1d


#NETWORK SETUP
LR = 1e-4 # learning rate
MOM = 0.9 # momentum
NUM_ITER = 2000 # number iterations
WD = 1 # weight decay for l2-regularization
Z_NUM = 32 # input seed
NGF = 64 # number of filters per layer
nc = 1 #num channels in the net I/0
alpha = 1e-5 #learning rate of Lasso
alpha_tv = 1e-1 #TV parameter for net loss
LENGTH = 1024


#CUDA SETUP
CUDA = torch.cuda.is_available()
print("On GPU: ", CUDA)

if CUDA :
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor


#SIGNAL GENERATION
signal = s.gen_data(LENGTH, 0.5, 1)

target_range = np.array(range(400, 430)) #the range in the signal where the target appears
target_len = len(list(target_range))

signal[target_range] += 0.04 #add artificial target to the background signal

signal += np.random.normal(loc = 0, scale = 0.0075, size=LENGTH) #add background noise
signal = inverse_utils.normalise(signal) #normalise signal to range [-1, 1]

x = np.zeros((LENGTH, 1))
x[:, 0] = signal

"""
plt.figure()
plt.plot(range(LENGTH), signal)
plt.title("Original Signal")
plt.show()
"""


#IMPUTATION SETUP
missing_samples = range(390, 420)
kept_samples = [x for x in range(LENGTH) if x not in missing_samples]
A = np.identity(LENGTH)[kept_samples, :] #the dropout transformation

y = x[kept_samples]

#x_hat = dip_utils.run_DIP_short(A, y, dtype, NGF = NGF, LR=LR, MOM=MOM, WD=WD, output_size=LENGTH, num_measurements=len(kept_samples), CUDA=CUDA, num_iter=NUM_ITER, alpha_tv=alpha_tv)

#A = spfft.idct(np.identity(LENGTH), norm='ortho', axis=0)[kept_samples, :]
#x_hat_lasso = inverse_utils.run_Lasso(A=A, y=y, output_size=LENGTH, alpha=1e-5)

x_missing = x.copy()
x_missing[missing_samples] = None

x_hat_gaussian = x.copy()
x_hat_gaussian[missing_samples] = np.random.normal(np.mean(y),np.std(y),(len(missing_samples), 1))

f_interp = interp1d(x = kept_samples, y = x[kept_samples].squeeze() + 2, kind='linear')
interped = (f_interp(missing_samples) - 2)
x_hat_interp = x.copy()
x_hat_interp[missing_samples] = interped.reshape(-1, 1)

plt.figure()
#plt.plot(range(LENGTH), x_hat, label = "Imputed")
#plt.plot(range(LENGTH), x_hat_lasso, label = "Imputed")
#plt.plot(range(LENGTH), x, label = "Original")
#plt.plot(range(LENGTH), x_hat_gaussian, label = "Gaussian-filled")
plt.plot(range(LENGTH), x_hat_interp, label = "Linear Interpolation")
plt.legend()
plt.grid(True)
plt.show()


"""
plt.figure()
plt.plot(range(LENGTH), x_missing, label = "W/Missing Values", color = 'green')
plt.legend()
plt.grid(True)
plt.show()
"""

#orig_normalised = s.two_pass_filtering(x, 20, target_len+5, 1.5)
#imputed_normalised = s.two_pass_filtering(x_hat, 20, target_len+5, 1.5)
#lasso_normalised = s.two_pass_filtering(x_hat_lasso, 20, target_len+5, 1.5)
#gaussian_normalised = s.two_pass_filtering(x_hat_gaussian, 20, target_len+5, 1.5)
interp_normalised = s.two_pass_filtering(x_hat_interp, 20, target_len+5, 1.5)

plt.figure()
#plt.plot(range(LENGTH), imputed_normalised, label = "Imputed")
#plt.plot(range(LENGTH), lasso_normalised, label = "LASSO", color='r')
#plt.plot(range(LENGTH), orig_normalised, label = "Original")
#plt.plot(range(LENGTH), gaussian_normalised, label = "Gaussian-filled", color = 'k')
plt.plot(range(LENGTH), interp_normalised, label = "Linear Interpolation", color = 'm')
plt.legend()
plt.grid(True)
plt.show()
