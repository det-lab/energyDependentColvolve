import numpy as np
import functions as fn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#### Double Gaussian Plot ####
x = np.linspace(-100, 100, 200)

convolvedSigKernArray = fn.convolution_2d_changing_kernel(fn.sigFunc(x),fn.kernFunc,x,2)

mathmaticaSolutionArray = fn.mathmaticaSolution(x)

test = fn.varGauss(max(fn.sArray(x))*1/np.diff(x)[0])

plt.plot(x, mathmaticaSolutionArray, color="red") # mathamatica analytical solution
plt.plot(x,convolvedSigKernArray, color="black") # python matrix method
plt.title("Plot of mathematicaSolution(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.xlim(-20,20)

plt.grid(True)
plt.show()

