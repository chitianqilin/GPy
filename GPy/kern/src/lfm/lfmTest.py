import lfm_C
import numpy as np
import sys

def lfmcomputeUpsilonMatrix(gamma1_p, sigma2, t1, t2):
    return lfm_C.computeUpsilonMatrix(gamma1_p, sigma2, t1.astype(np.float64), t2.astype(np.float64))

if __name__ == '__main__':
    gamma1_p = 5 + 0.5j
    sigma2 = 1
    t1 = np.arange(4)
    t2 = np.arange(3)
    result = lfmcomputeUpsilonMatrix(gamma1_p, sigma2, t1, t2)
    print(result)