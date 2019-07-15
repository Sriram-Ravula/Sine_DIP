from scipy.special import gamma, kn
import numpy as np


def __get_k(x, b, v):
    one = 2 * b / gamma(v)
    two = (b * x / 2) ** v
    three = kn(v - 1, b * x)

    return one * two * three

def gen_data(n, b, v):
    pdf = np.zeros(n)

    for i in range(n):
        j = i + int(n / 5)

        pdf[i] = __get_k(j / 100, b, v)

    return pdf


def scale(x):
    mins = np.min(x)
    maxes = np.max(x)

    return (x - mins) / (maxes - mins)


def __single_pass(x, m, Ws, Wg):
    n = len(list(x.squeeze()))

    upper = int(m + (Ws + (Wg - 1) / 2))
    lower = int(m - (Ws + (Wg - 1) / 2))

    inner_range = upper - lower + 1

    z = 0

    for j in range(inner_range):
        i = j + lower

        if (abs(i - m) > (Wg - 1) / 2):

            if (i >= n):
                z += x[n - 1]
            elif (i < 0):
                z += x[0]
            else:
                z += x[i]

    z = z / (2 * Ws)

    return z


def __clip(x, z, m, r):
    if (x[m] < r * z[m]):
        return x[m]
    else:
        return z[m]


def two_pass_filtering(x, Ws, Wg, r):
    n = len(list(x.squeeze()))

    z = np.zeros(n)

    for i in range(n):
        z[i] = __single_pass(x, i, Ws, Wg)

    y = np.zeros(n)

    for i in range(n):
        y[i] = __clip(x, z, i, r)

    y_hat = np.zeros(n)

    for i in range(n):
        y_hat[i] = __single_pass(y, i, Ws, Wg)

    x = np.abs(x)
    y_hat = np.abs(y_hat)

    return np.divide(x.squeeze(), y_hat.squeeze())