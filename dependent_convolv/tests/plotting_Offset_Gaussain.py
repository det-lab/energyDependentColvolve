import numpy as np
import dependent_convolv.functions.convolv as fn
import matplotlib.pyplot as plt
from scipy.integrate import quad
#### Double Gaussian Plot ####
x = np.linspace(-100, 100, 2000)

convolvedSigKernArray = fn.convolution_2d_changing_kernel(fn.sigFuncPositiveOffset(x),fn.kernFunc,x)

def plotting():

    plt.plot(x, fn.sigFuncPositiveOffset(x), color = "blue", label = "Signal func")
    plt.plot(x,convolvedSigKernArray,linewidth=2, color="black", label = "Algorithm Computation") # python matrix method
    plt.title("Plot of Mathematica Solution and Algorithm Computation")
    plt.legend(loc='upper right')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.xlim(-5,25)

    plt.grid(True)
    plt.show()

plotting()