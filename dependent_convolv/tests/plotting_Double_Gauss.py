import numpy as np
import dependent_convolv.functions.convolv as fn
import matplotlib.pyplot as plt
from scipy.integrate import quad
#### Double Gaussian Plot ####
x = np.linspace(-100, 100, 2000)

convolvedSigKernArray = fn.convolution_2d_changing_kernel(fn.sigFunc(x),fn.kernFunc,x)

mathmaticaSolutionArray = fn.mathmatica_double_gauss(x)

def plotting():

    y15_a = fn.kernFunc(-15+x, -15)
    y15_b = fn.kernFunc(15+x, 15)
    y0  = fn.kernFunc(x,0)


    plt.plot(x, fn.sigFunc(x), color = "blue", label = "Signal func")
    #plt.plot(x, y15_a, color = "red", label = "Kernel func at x=-15")
    #plt.plot(x, y15_b, color = "green", label = "Kernel func at x=15")
    #plt.plot(x, y0, color = "blue", label = "Kernel func at x=0")
    #plt.fill_between(x, y0, alpha=0.3, color='blue', label="Area = 1")
    #plt.fill_between(x, y15_a, alpha=0.3, color='red', label="Area = 1")
    #plt.fill_between(x, y15_b, alpha=0.3, color='green', label="Area = 1")
    plt.plot(x, mathmaticaSolutionArray,linewidth=5, color="red", label = "Mathematica Solution") # mathamatica analytical solution
    plt.plot(x,convolvedSigKernArray,linewidth=2, color="black", label = "Algorithm Computation") # python matrix method
    plt.title("Plot of Mathematica Solution and Algorithm Computation")
    plt.legend(loc='upper right')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.xlim(-20,20)

    plt.grid(True)
    plt.show()

plotting()