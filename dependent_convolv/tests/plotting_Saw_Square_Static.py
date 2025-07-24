import numpy as np
import matplotlib.pyplot as plt
import dependent_convolv.functions.convolv as fn

##### Saw and Square Case #####
x = np.linspace (-20,20,10000)

mathmaticaSol = fn.mathamatica_Saw_Square(x) #Mathematica's analytical solution

numpySol = np.convolve(fn.sawWave1(x), fn.squareWave1(x), mode="same") * np.diff(x)[0] #numpys standard convolve

convolvedSigKernArray = fn.convolution_2d_changing_kernel(fn.sawWave1(x), fn.squareWave1, x) #calls custom convolve func

def plotting():
    print(max(abs(mathmaticaSol - convolvedSigKernArray)))
    print(max(abs(mathmaticaSol - numpySol)))
    #creating plot for all three generated arrays vs x axis
    plt.plot(x,convolvedSigKernArray, linewidth=8, color = "blue", label = "Python Algorithm") 
    plt.plot(x,numpySol, linewidth=4,color = "yellow", label="Numpy Convolution")
    plt.plot(x,mathmaticaSol, linewidth=1, color="red", label = "Mathmatica Soultion")
    plt.title("Plot of mathematicaSolution(x)")
    plt.legend(loc='upper right')
    plt.xlabel("Energy (E)")
    plt.ylabel("f(E)")
    plt.xlim(-5,10)
    plt.grid(True)
    plt.show()

plotting()