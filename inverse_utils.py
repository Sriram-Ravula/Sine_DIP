import random
import torch
import numpy as np
import math
import scipy.fftpack as spfft
import wavio
from sklearn.linear_model import Lasso


def read_wav(filename):
    """
    Reads a wav file and returns the associated data

    Parameters
    ----------
    filename : string
        The name of the wav file you want to read

    Returns
    -------
    rate
        The sampling rate of the wav
    length
        The number of samples in each channel
    resolution
        The number of bytes per sample
    nc
        The number of sound channels
    x
        [length, nc]-dimension array containing the wav data
    """

    wave = wavio.read(filename)
    rate = wave.rate
    length = wave.data.shape[0]
    resolution = wave.sampwidth
    nc = wave.data.shape[1]
    x = wave.data

    return [rate, length, resolution, nc, x]

def normalise(x, bits):
    """
    Normalizes an input array to have range [-1, 1]

    Parameters
    ----------
    x : int or double array
        The array of data to normalise
    bits: int
        The bit resolution of x. We use 2^(bits - 1) to normalise range

    Returns
    -------
    x_normed
        The data, range-normalised to [-1, 1]
    """

    return x/(2**(bits-1))

    #mu = np.mean(np.squeeze(x))
    #sigma = np.std(np.squeeze(x))

    #return (x - mu)/sigma

def heart_normalise(x):
    x0 = np.squeeze(x)
    max = np.amax(x0)
    min = np.amin(x0)
    range = max-min
    y = 2*(x - min)/range - 1
    return y

#Renormalises array to have +/- 2^(bits-1) range
def renormalise(x, bits):

    return x*(2**(bits-1))

#Generates a size [num_samples, nc] array of Gaussian noise from N(0, std^2)
def get_noise(num_samples = 16384, nc = 1, std = 0.1):
    return (std * np.random.randn(num_samples, nc))

#Generates the sampling matrix phi (for DIP) and measurement matrix A = phi*psi (where psi is the IDCT matrix for sparse reconstruction, e.g. Lasso)
def get_A(case, num_measurements = 1000, original_length = 16384):

    if case == 'Dropout':
        kept_samples = random.sample(range(0, original_length), num_measurements)

        A = spfft.idct(np.identity(original_length), norm='ortho', axis=0)[kept_samples, :]
        phi = np.eye(original_length)[kept_samples, :]

        return [phi, A]

    if case =='CS':
        phi = (1 / math.sqrt(1.0 * num_measurements)) * np.random.randn(num_measurements, original_length)
        A = np.matmul(phi, spfft.idct(np.identity(original_length), norm='ortho', axis=0))

        return [phi, A]

    if case == 'Deconvolution':
        #FIX THIS SECTION
        phi = np.eye(original_length)
        A = spfft.idct(np.identity(original_length), norm='ortho', axis=0)

        return[phi, A]

    if case == 'Identity':
        phi = np.eye(original_length)
        A = spfft.idct(np.identity(original_length), norm='ortho', axis=0)

        return [phi, A]

    else:
        print("WRONG INPUT TO get_A: INVALID CASE. TRY AGAIN")

        exit(0)

#Run Lasso reconstruction given measurement matrix A and observed measurements y
def run_Lasso(A, y, output_size = 16834, alpha = 0.00001):
    lasso = Lasso(alpha=alpha)
    lasso.fit(A, y)

    x_hat = np.array(lasso.coef_).reshape(output_size)
    x_hat = spfft.idct(x_hat, norm='ortho', axis=0)
    x_hat = x_hat.reshape(-1, 1)

    return x_hat
