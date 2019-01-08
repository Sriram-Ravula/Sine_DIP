import matplotlib.pyplot as plt
import inverse_utils
import numpy as np

load_loc = "/home/sravula/Projects/compsensing_dip-master/Results/"
test_type = "CS"
sample = "beat"

filename = load_loc + test_type + "/" + sample + "/" + sample + "-"

num_measurements = inverse_utils.read_log(filename + "NLM-VAMP.txt")[:,0]

results_DIP = inverse_utils.read_log(filename + "DIP.txt")[:,1]
results_Lasso = inverse_utils.read_log(filename + "Lasso.txt")[:,1]
results_Wiener = inverse_utils.read_log(filename + "Wiener-VAMP.txt")[:,1]
results_NLM = inverse_utils.read_log(filename + "NLM-VAMP.txt")[:,1]

plt.figure()
plt.plot(num_measurements, results_Lasso, label="Lasso", color='b', marker = 'D')
plt.plot(num_measurements, results_Wiener, label="Wiener-VAMP", color = 'g', marker = '+')
plt.plot(num_measurements, results_NLM, label="NLM-VAMP", color = 'k', marker = '^')
plt.plot(num_measurements, results_DIP, label="DIP", color='r', marker = 'o')
plt.xlabel("Num Measurements")
plt.ylabel("MSE")
plt.title("Beat - Compressed Sensing")
plt.ylim(0, 0.03)
plt.legend()
plt.show()


