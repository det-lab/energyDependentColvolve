import numpy as np
import matplotlib.pyplot as plt
import dependent_convolv.functions.convolv as fn

#### Square and Sin wave plot ####
x = np.linspace(-20,20,10000)

mathSol = fn.mathematica_Square_Sin(x)

convolvedSigKernArray = fn.convolution_2d_changing_kernel(fn.squareWave2(x), fn.sinWave, x)

numpySol = np.convolve(fn.squareWave2(x), fn.sinWave(x), mode="same") * np.diff(x)[0] #note that using numpys convolve, must be normalized by dx

def plotting():
    print(max(abs(mathSol - convolvedSigKernArray)))
    plt.plot(x, convolvedSigKernArray,linewidth=8, color= "blue", label="Python Algorithm")
    plt.plot(x, numpySol, linewidth = 4, color = "yellow", label = "Numpy Solution")
    plt.plot(x, mathSol,color = "red", label="Mathematica Solution")
    plt.title("Square and Sin wave convolution")
    plt.legend(loc='upper right')
    plt.xlabel("Energy (E)")
    plt.ylabel("f(E)")
    plt.xlim(-5,10)
    plt.grid(True)
    plt.show()

plotting()