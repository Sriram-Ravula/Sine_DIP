import torch
import numpy as np
import matplotlib.pyplot as plt
import inverse_utils
import dip_utils
import time
import pywt

LR = 1e-4 # learning rate
MOM = 0.9 # momentum
NUM_ITER = 1000 # number iterations
WD = 1 # weight decay for l2-regularization
Z_NUM = 32 # input seed
NGF = 64 # number of filters per layer
nc = 1 #num channels in the net I/0
alpha = 1e-5 #learning rate of Lasso
alpha_tv = [0, 1e-4, 1e-3, 1e-2, 1e-1]#TV parameter for net loss
LENGTH = 1024


CUDA = torch.cuda.is_available()
print("On GPU: ", CUDA)

if CUDA :
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor


test_type = "DCT" #Imputation, CS, DCT, or Denoising

#x0 = np.zeros((LENGTH, 1))
#x0[:,0] = pywt.data.demo_signal(name='twochirp', n=LENGTH)
#x = inverse_utils.normalise(x0)

data_loc = "/home/sravula/AirQualityUCI/AirQuality.csv" #Location of the air quality CSV data
sample = "O3-1"  #Choice of: O3-1, O3-2, NO2-1, NO2-2, CO-1, or CO-2

x0 = inverse_utils.get_air_data(loc=data_loc, data=sample, length=LENGTH)
x = np.zeros((LENGTH, 1))
x[:, 0] = inverse_utils.normalise(x0)


if test_type == "Imputation" or test_type == "CS" or test_type == "DCT":
    num_measurements = [10, 25, 50, 100, 200]
elif test_type == "Denoising":
    NUM_ITER = 300 #reduce iterations for denoising to prevent overfitting
    num_measurements = [LENGTH]
    std = 0.1  # the standard deviation of AWGN, if denoising
    noise = inverse_utils.get_noise(num_samples=LENGTH, nc = nc, std = std)
    x_orig = x.copy()
    x = x+noise
else:
    print("UNSUPPORTED TEST TYPE. PLEASE CHOOSE: Imputation, CS, DCT, OR Denoising")
    exit(0)


mse_list = np.zeros((len(num_measurements), 5))

start = time.time()
for i in range(len(num_measurements)):

    if test_type != "Imputation": #if the test is not imputation, we do not get a list of kept samples
        phi, A = inverse_utils.get_A(case=test_type, num_measurements=num_measurements[i], original_length=LENGTH)
    else:
        phi, A, kept_samples = inverse_utils.get_A(case=test_type, num_measurements=num_measurements[i], original_length=LENGTH)
        imputed_samples = [z for z in range(0, LENGTH) if z not in kept_samples]

    y = np.dot(phi, x)  #create the measurements

    num_instances = 10 #how many instances of lasso and DIP we wish to run to average results

    for t in range(num_instances):

        x_hat_DIP = dip_utils.run_DIP_short(phi, y, dtype, NGF = NGF, LR=LR, MOM=MOM, WD=WD, output_size=LENGTH, num_measurements=num_measurements[i], CUDA=CUDA, num_iter=NUM_ITER, alpha_tv=alpha_tv[0])

        if test_type == "Imputation": #for imputation, we only wish to calculate MSE on the imputed values
            mse_list[i, 0] = mse_list[i, 0] + np.mean((np.squeeze(x_hat_DIP)[imputed_samples] - np.squeeze(x)[imputed_samples])**2)
        elif test_type == "Denoising":
            mse_list[i, 0] = mse_list[i, 0] + np.mean((np.squeeze(x_hat_DIP) - np.squeeze(x_orig))**2)
        else:
            mse_list[i, 0] = mse_list[i, 0] + np.mean((np.squeeze(x_hat_DIP) - np.squeeze(x)) ** 2)


        x_hat_DIP = dip_utils.run_DIP_short(phi, y, dtype, NGF = NGF, LR=LR, MOM=MOM, WD=WD, output_size=LENGTH, num_measurements=num_measurements[i], CUDA=CUDA, num_iter=NUM_ITER, alpha_tv=alpha_tv[1])

        if test_type == "Imputation": #for imputation, we only wish to calculate MSE on the imputed values
            mse_list[i, 1] = mse_list[i, 1] + np.mean((np.squeeze(x_hat_DIP)[imputed_samples] - np.squeeze(x)[imputed_samples])**2)
        elif test_type == "Denoising":
            mse_list[i, 1] = mse_list[i, 1] + np.mean((np.squeeze(x_hat_DIP) - np.squeeze(x_orig))**2)
        else:
            mse_list[i, 1] = mse_list[i, 1] + np.mean((np.squeeze(x_hat_DIP) - np.squeeze(x)) ** 2)


        x_hat_DIP = dip_utils.run_DIP_short(phi, y, dtype, NGF = NGF, LR=LR, MOM=MOM, WD=WD, output_size=LENGTH, num_measurements=num_measurements[i], CUDA=CUDA, num_iter=NUM_ITER, alpha_tv=alpha_tv[2])

        if test_type == "Imputation": #for imputation, we only wish to calculate MSE on the imputed values
            mse_list[i, 2] = mse_list[i, 2] + np.mean((np.squeeze(x_hat_DIP)[imputed_samples] - np.squeeze(x)[imputed_samples])**2)
        elif test_type == "Denoising":
            mse_list[i, 2] = mse_list[i, 2] + np.mean((np.squeeze(x_hat_DIP) - np.squeeze(x_orig))**2)
        else:
            mse_list[i, 2] = mse_list[i, 2] + np.mean((np.squeeze(x_hat_DIP) - np.squeeze(x)) ** 2)


        x_hat_DIP = dip_utils.run_DIP_short(phi, y, dtype, NGF = NGF, LR=LR, MOM=MOM, WD=WD, output_size=LENGTH, num_measurements=num_measurements[i], CUDA=CUDA, num_iter=NUM_ITER, alpha_tv=alpha_tv[3])

        if test_type == "Imputation": #for imputation, we only wish to calculate MSE on the imputed values
            mse_list[i, 3] = mse_list[i, 3] + np.mean((np.squeeze(x_hat_DIP)[imputed_samples] - np.squeeze(x)[imputed_samples])**2)
        elif test_type == "Denoising":
            mse_list[i, 3] = mse_list[i, 3] + np.mean((np.squeeze(x_hat_DIP) - np.squeeze(x_orig))**2)
        else:
            mse_list[i, 3] = mse_list[i, 3] + np.mean((np.squeeze(x_hat_DIP) - np.squeeze(x)) ** 2)


        x_hat_DIP = dip_utils.run_DIP_short(phi, y, dtype, NGF = NGF, LR=LR, MOM=MOM, WD=WD, output_size=LENGTH, num_measurements=num_measurements[i], CUDA=CUDA, num_iter=NUM_ITER, alpha_tv=alpha_tv[4])

        if test_type == "Imputation": #for imputation, we only wish to calculate MSE on the imputed values
            mse_list[i, 4] = mse_list[i, 4] + np.mean((np.squeeze(x_hat_DIP)[imputed_samples] - np.squeeze(x)[imputed_samples])**2)
        elif test_type == "Denoising":
            mse_list[i, 4] = mse_list[i, 4] + np.mean((np.squeeze(x_hat_DIP) - np.squeeze(x_orig))**2)
        else:
            mse_list[i, 4] = mse_list[i, 4] + np.mean((np.squeeze(x_hat_DIP) - np.squeeze(x)) ** 2)


    mse_list[i,:] = mse_list[i,:]/num_instances


    print("\n" + str(num_measurements[i]) + " Measurements - " + str(alpha_tv[0]) + " mse: " + str(mse_list[i, 0]))
    print(str(num_measurements[i]) + " Measurements - " + str(alpha_tv[1]) + " mse: " + str(mse_list[i, 1]))
    print(str(num_measurements[i]) + " Measurements - " + str(alpha_tv[2]) + " mse: " + str(mse_list[i, 2]))
    print(str(num_measurements[i]) + " Measurements - " + str(alpha_tv[3]) + " mse: " + str(mse_list[i, 3]))
    print(str(num_measurements[i]) + " Measurements - " + str(alpha_tv[4]) + " mse: " + str(mse_list[i, 4]))

    end = time.time()
    print("\n" + "Execution Time: ", round(end - start, 2), "seconds")


plt.figure()
plt.plot(num_measurements, mse_list[:, 0], label = str(alpha_tv[0]), color = 'r', marker = 'o')
plt.plot(num_measurements, mse_list[:, 1], label = str(alpha_tv[1]), color = 'b', marker = 'D')
plt.plot(num_measurements, mse_list[:, 2], label = str(alpha_tv[2]), color = 'g', marker = '+')
plt.plot(num_measurements, mse_list[:, 3], label = str(alpha_tv[3]), color = 'k', marker = '^')
plt.plot(num_measurements, mse_list[:, 4], label = str(alpha_tv[4]), color = 'y', marker = 's')
plt.xlabel("Num Measurements")
plt.ylabel("MSE")
plt.title("Alpha_TV Param Comp")
plt.legend()
plt.show()


