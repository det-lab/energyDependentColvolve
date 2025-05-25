import numpy as np
import matplotlib.pyplot as plt
import functions as fn

##### Saw and Square Case #####
x = np.linspace (-20,20,10000)

mathmaticaSol = fn.mathamatica_Saw_Square(x) #Mathematica's analytical solution

numpySol = np.convolve(fn.sawWave1(x), fn.squareWave1(x), mode="same") * np.diff(x)[0] #numpys standard convolve

convolvedSigKernArray = fn.convolution_2d_changing_kernel(fn.sawWave1(x), fn.squareWave1, x) #calls custom convolve func

def plotting():
    #creating plot for all three generated arrays vs x axis
    plt.plot(x,convolvedSigKernArray,linewidth=5, color = "black") 
    plt.plot(x,numpySol,linewidth=3,color = "blue")
    plt.plot(x,mathmaticaSol, color="red")
    plt.title("Plot of mathematicaSolution(x)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.xlim(-20,20)

    plt.grid(True)
    plt.show()


print(max(abs(convolvedSigKernArray-mathmaticaSol)))