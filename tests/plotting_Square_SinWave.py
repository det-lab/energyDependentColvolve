import numpy as np
import matplotlib.pyplot as plt
import functions as fn

#### Square and Sin wave plot ####
x = np.linspace(-20,20,1000)

mathSol = fn.mathematica_Square_Sin(x)

convolvedSigKernArray = fn.convolution_2d_changing_kernel(fn.squareWave2(x), fn.sinWave, x)

numpySol = np.convolve(fn.squareWave2(x), fn.sinWave(x), mode="same") * np.diff(x)[0] #note that using numpys convolve, must be normalized by dx

def plotting():

    plt.plot(x, convolvedSigKernArray,linewidth=5, color= "black")
    plt.plot(x, numpySol, linewidth = 3, color = "blue")
    plt.plot(x, mathSol,color = "red")
    plt.show()