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
import pandas as pd

#########################
LENGTH = 1024
LR = 1e-4
MOM = 0.9 # momentum
NUM_ITER = 3000 # number iterations
WD = 1
Z_NUM = 32 # input seed
NGF = [16, 32, 64, 96, 128] # number of filters per layer
nc = 1 #num channels in the net I/0
alpha = 1e-5 #learning rate of Lasso
alpha_tv = 1e-1
#########################

x0 = inverse_utils.get_air_data(loc = "/home/sravula/AirQualityUCI/AirQuality.csv")
x = np.zeros((LENGTH, 1))
x[:,0] = inverse_utils.normalise(x0)


CUDA = torch.cuda.is_available()
print("On GPU: ", CUDA)

if CUDA :
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor
#########################

filename = "Heart1"
test_type = "CS"


if test_type == 'Imputation' or test_type =='CS':
    num_measurements = [10, 25, 50, 75, 150]
else:
    num_measurements = [LENGTH]

mse_list = np.zeros((len(num_measurements), 5))
lasso_mse = np.zeros((len(num_measurements), 1))
##########################

start = time.time()
for i in range(len(num_measurements)):

    if test_type != "Imputation": #if the test is not imputation, we do not get a list of kept samples
        phi, A = inverse_utils.get_A(case=test_type, num_measurements=num_measurements[i], original_length=LENGTH)
    else:
        phi, A, kept_samples = inverse_utils.get_A(case=test_type, num_measurements=num_measurements[i], original_length=LENGTH)

    y = np.dot(phi, x)  #create the measurements

    num_instances = 5 #how many instances of lasso and DIP we wish to run to average results

    mseone = 0
    msetwo = 0
    msethree = 0
    msefour = 0
    msefive = 0

    mse_lasso = 0
    for t in range(num_instances):

        x_hat_lasso = inverse_utils.run_Lasso(A=A, y=y, output_size=LENGTH, alpha=alpha)

        if test_type == "Imputation": #for imputation, we want to add back in the values we know to the predictions
            removed_samples = [z for z in range(0, LENGTH) if z not in kept_samples]
            mse_lasso = mse_lasso + np.mean((np.squeeze(x_hat_lasso)[removed_samples] - np.squeeze(x)[removed_samples])**2)

        else:
            mse_lasso = mse_lasso + np.mean((np.squeeze(x_hat_lasso) - np.squeeze(x))**2)


        x_hat_DIP = dip_utils.run_DIP_heart(phi, y, dtype, NGF = NGF[0], LR=LR, MOM=MOM, WD=WD, output_size=LENGTH, num_measurements=num_measurements[i], CUDA=CUDA, num_iter=NUM_ITER, alpha_tv=alpha_tv)

        if test_type == "Imputation": #for imputation, we want to add back in the values we know to the predictions
            mseone = mseone + np.mean((np.squeeze(x_hat_DIP)[removed_samples] - np.squeeze(x)[removed_samples])**2)

        else:
            mseone = mseone + np.mean((np.squeeze(x_hat_DIP) - np.squeeze(x))**2)

        x_hat_DIP = dip_utils.run_DIP_heart(phi, y, dtype, NGF = NGF[1], LR=LR, MOM=MOM, WD=WD, output_size=LENGTH, num_measurements=num_measurements[i], CUDA=CUDA, num_iter=NUM_ITER, alpha_tv=alpha_tv)

        if test_type == "Imputation": #for imputation, we want to add back in the values we know to the predictions
            msetwo = msetwo + np.mean((np.squeeze(x_hat_DIP)[removed_samples] - np.squeeze(x)[removed_samples])**2)

        else:
            msetwo = msetwo + np.mean((np.squeeze(x_hat_DIP) - np.squeeze(x))**2)

        x_hat_DIP = dip_utils.run_DIP_heart(phi, y, dtype, NGF = NGF[2], LR=LR, MOM=MOM, WD=WD, output_size=LENGTH, num_measurements=num_measurements[i], CUDA=CUDA, num_iter=NUM_ITER, alpha_tv=alpha_tv)

        if test_type == "Imputation": #for imputation, we want to add back in the values we know to the predictions
            msethree = msethree+ np.mean((np.squeeze(x_hat_DIP)[removed_samples] - np.squeeze(x)[removed_samples])**2)

        else:
            msethree = msethree + np.mean((np.squeeze(x_hat_DIP) - np.squeeze(x))**2)

        x_hat_DIP = dip_utils.run_DIP_heart(phi, y, dtype, NGF = NGF[3], LR=LR, MOM=MOM, WD=WD, output_size=LENGTH, num_measurements=num_measurements[i], CUDA=CUDA, num_iter=NUM_ITER, alpha_tv=alpha_tv)

        if test_type == "Imputation": #for imputation, we want to add back in the values we know to the predictions
            msefour = msefour + np.mean((np.squeeze(x_hat_DIP)[removed_samples] - np.squeeze(x)[removed_samples])**2)

        else:
            msefour = msefour + np.mean((np.squeeze(x_hat_DIP) - np.squeeze(x))**2)

        x_hat_DIP = dip_utils.run_DIP_heart(phi, y, dtype, NGF = NGF[4], LR=LR, MOM=MOM, WD=WD, output_size=LENGTH, num_measurements=num_measurements[i], CUDA=CUDA, num_iter=NUM_ITER, alpha_tv=alpha_tv)

        if test_type == "Imputation": #for imputation, we want to add back in the values we know to the predictions
            msefive = msefive + np.mean((np.squeeze(x_hat_DIP)[removed_samples] - np.squeeze(x)[removed_samples])**2)

        else:
            msefive = msefive + np.mean((np.squeeze(x_hat_DIP) - np.squeeze(x))**2)

    mseone = mseone/float(num_instances)
    msetwo = msetwo / float(num_instances)
    msethree = msethree / float(num_instances)
    msefour = msefour / float(num_instances)
    msefive = msefive / float(num_instances)
    mse_lasso = mse_lasso/float(num_instances)

    print("Done with ", num_measurements[i])

    mse_list[i, 0] = mseone
    mse_list[i, 1] = msetwo
    mse_list[i, 2] = msethree
    mse_list[i, 3] = msefour
    mse_list[i, 4] = msefive

    lasso_mse[i, 0] = mse_lasso

end = time.time()
print("Execution Time: ", round(end-start, 2), "seconds")

plt.figure()
plt.plot(num_measurements, mse_list[:, 0], label = "16", color = 'r', marker = 'o')
plt.plot(num_measurements, mse_list[:, 1], label = "32", color = 'b', marker = 'D')
plt.plot(num_measurements, mse_list[:, 2], label = "64", color = 'g', marker = '+')
plt.plot(num_measurements, mse_list[:, 3], label = "96", color = 'k', marker = '^')
plt.plot(num_measurements, mse_list[:, 4], label = "128", color = 'y', marker = 's')
plt.plot(num_measurements, lasso_mse[:, 0], label = "LASSO", color = 'm', marker = "o")
plt.xlabel("Num Measurements")
plt.ylabel("MSE")
plt.title(filename + "-" + test_type  + " LR parameter comparison")
plt.legend()
#plt.savefig(save_loc + filename + "-" + test_type + noise_str + "- MOM_comp.jpg")
plt.show()