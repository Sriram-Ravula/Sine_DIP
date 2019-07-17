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
NUM_ITER = 3000 # number iterations
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
target_range = np.union1d(target_range, np.array(range(800, 830)))
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
missing_samples = np.array(range(395, 425))
missing_samples = np.union1d(missing_samples, np.array(range(805, 835)))
missing_samples = np.union1d(missing_samples, np.array(range(200, 230)))

kept_samples = [x for x in range(LENGTH) if x not in missing_samples]

A = np.identity(LENGTH)[kept_samples, :] #the dropout transformation

y = x[kept_samples]

#DIP imputation
#x_hat = dip_utils.run_DIP_short(A, y, dtype, NGF = NGF, LR=LR, MOM=MOM, WD=WD, output_size=LENGTH, num_measurements=len(kept_samples), CUDA=CUDA, num_iter=NUM_ITER, alpha_tv=alpha_tv)

#Lasso imputation
phi = spfft.idct(np.identity(LENGTH), norm='ortho', axis=0)[kept_samples, :]
x_hat_lasso = inverse_utils.run_Lasso(A=phi, y=y, output_size=LENGTH, alpha=1e-5)

#Signal with missing values
x_missing = x.copy()
x_missing[missing_samples] = None

#Gaussian imputation
x_hat_gaussian = x.copy()
x_hat_gaussian[missing_samples] = np.random.normal(np.mean(y), 0.1, (len(missing_samples), 1))

#Linear interpolation
f_interp = interp1d(x = kept_samples, y = x[kept_samples].squeeze() + 2, kind='linear')
interped = (f_interp(missing_samples) - 2)
x_hat_interp = x.copy()
x_hat_interp[missing_samples] = interped.reshape(-1, 1)

#Plotting signals before imputation
plt.figure()
#plt.plot(range(LENGTH), x_hat, label = "DIP Imputation", color = 'b')
plt.plot(range(LENGTH), x_hat_lasso, label = "Imputed", color = 'g')
#plt.plot(range(LENGTH), x, label = "Original", color = 'r')
#plt.plot(range(LENGTH), x_hat_gaussian, label = "Gaussian-filled", color = 'c')
#plt.plot(range(LENGTH), x_hat_interp, label = "Linear Interpolation", color = 'm')
#plt.plot(range(LENGTH), x_missing, label = "With Missing Values", color = 'k')
plt.legend()
plt.grid(True)
plt.title("Deep Image Prior")
plt.show()

#Normalize imputed signals
#orig_normalised = s.two_pass_filtering(x+2, 20, 35, 1)
#imputed_normalised = s.two_pass_filtering(x_hat+2, 20, 35, 1)
lasso_normalised = s.two_pass_filtering(x_hat_lasso + 2, 20, 35, 1)
#gaussian_normalised = s.two_pass_filtering(x_hat_gaussian + 2, 20, 35, 1)
#interp_normalised = s.two_pass_filtering(x_hat_interp + 2, 20, 35, 1)

plt.figure()
#plt.plot(range(LENGTH), imputed_normalised, label = "DIP Imputation", color = 'b')
plt.plot(range(LENGTH), lasso_normalised, label = "Lasso", color = 'g')
#plt.plot(range(LENGTH), orig_normalised, label = "Original", color = 'r')
#plt.plot(range(LENGTH), gaussian_normalised, label = "Gaussian-filled", color = 'c')
#plt.plot(range(LENGTH), interp_normalised, label = "Linear Interpolation", color = 'm')
plt.ylim(0.5, 1.5)
plt.legend()
plt.grid(True)
plt.show()