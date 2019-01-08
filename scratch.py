import inverse_utils
import numpy as np

# A = inverse_utils.get_A("Dropout", num_measurements=5, original_length=10)[0]
# inverse_utils.save_matrix(A, "/home/sravula/Projects/A.mat")

# inverse_utils.save_log("test", "Denoising", "Net", 5, "/home/sravula/Projects/gol.txt")

# a = np.zeros((3,2))
# a[0,0] = 1
# a[0,1] = 10
# a[1,0] = 2
# a[1,1] = 20
# a[2,0] = 3
# a[2,1] = 30
#
# inverse_utils.save_log("test", "Imputation", "Net", a, "/home/sravula/Projects/log.txt")

# a = inverse_utils.read_log("/home/sravula/Projects/log.txt")
# print(a)

b = inverse_utils.read_log("/home/sravula/Projects/gol.txt")
print(b)