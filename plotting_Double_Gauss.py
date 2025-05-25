import numpy as np
import functions as fn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#### Double Gaussian Plot ####
x = np.linspace(-100, 100, 2000)

convolvedSigKernArray = fn.convolution_2d_changing_kernel(fn.sigFunc(x),fn.kernFunc,x)

mathmaticaSolutionArray = fn.mathmatica_double_gauss(x)

test = fn.varGauss(max(fn.sArray(x))*1/np.diff(x)[0])

def plotting():

    temp1 = np.exp(-1*(-15-x)**2 / (2*(np.sqrt((15**2)/5 + 1))))
    temp2 = np.exp(-1*(15-x)**2 / (2*(np.sqrt((15**2)/5 + 1))))

    plt.plot(x, fn.sigFunc(x), color = "pink", label = "Signal func")
    plt.plot(x, fn.kernFunc(x, 0), color = "blue", label = "Kernel func at x=0")
    plt.plot(x, temp1, color = "green", label = "Kernel func at x=-15")
    plt.plot(x, temp2, color = "purple", label = "Kernel func at x=15")
    plt.plot(x, mathmaticaSolutionArray,linewidth=5, color="red", label = "Mathematica Solution") # mathamatica analytical solution
    plt.plot(x,convolvedSigKernArray,linewidth=2, color="black", label = "Algorithm Computation") # python matrix method
    plt.title("Plot of Mathematica Solution and Algorithm Computation")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.xlim(-20,20)

    plt.grid(True)
    plt.show()
