import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def convolution_2d_changing_kernel(signal, kernel, axis, argNum):
    #checks if fucntion args are 2
    if argNum == 2:
        kernelMatrix = []
        for x in axis: 
            kernelMatrix.append(kernel(x-axis,x))
        kernelMatrix = np.array(kernelMatrix)
    #checks if fucntion args are 1
    if argNum == 1:
        kernelMatrix = []
        for x in axis:
            kernelMatrix.append(kernel(x-axis))
        kernelMatrix = np.array(kernelMatrix)
    
     # Initialize the output array
    output_length = len(signal)
    output = np.zeros(output_length)

    #Normilization of output, however is much more irrelivent now due to the step amount being constant
    stepList = np.diff(axis)
    step = np.append(stepList, stepList[-1])
    # Dot product of the signal and kernel, the kernel is already moving along the signal with the 2d raw kernel
    for i, kernel in enumerate(kernelMatrix):
        
        #sums the the 1d arrays to return a single value to then be appended to the convolved array output
        output[i] = np.sum(signal * kernel * step)

    return output

##### Double gaussian ######

x = np.linspace(-100, 100, 200)

def mathmaticaSolution(x):
    numerator = 3 * np.exp(
        -1 * (np.sqrt(5) * x**2) / (2 * (9 * np.sqrt(5) + np.sqrt(5 + x**2)))
    ) * np.sqrt(2 * np.pi)

    denominator = np.sqrt(1 + (9 * np.sqrt(5)) / (np.sqrt(5 + x**2)))

    return numerator / denominator

mathmaticaSolutionArray = mathmaticaSolution(x)

def sigFunc(xAxis):
    return np.exp(-1 * (xAxis**2)/18)

def kernFunc(xArray, xVal):
    return np.exp(-1*(xArray)**2 / (2*(np.sqrt(xVal**2/5 + 1))))

### VarConvolve Github ###

#creates the width of gaussians at x values for varcon
def sArray(xVal):
    return np.sqrt((xVal**2) /5 + 1)

#generates the kernel gaussian for varcon
def varGauss(s):
    size_grid = int(s*4)
    grid = np.mgrid[-size_grid:size_grid+1]
    g = np.exp(-(grid**2/(2*float(s))))
    
    return g
### End ###

convolvedSigKernArray = convolution_2d_changing_kernel(sigFunc(x),kernFunc,x,2)

test = varGauss(max(sArray(x))*1/np.diff(x)[0])

plt.plot(x, mathmaticaSolutionArray, color="red") # mathamatica analytical solution
plt.plot(x,convolvedSigKernArray, color="black") # python matrix method
plt.title("Plot of mathematicaSolution(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.xlim(-20,20)

plt.grid(True)
plt.show()

##### Saw and Square Case #####

def f(x):
    return np.piecewise(x, 
                        [x < 0, (0 <= x) & (x < 2), x >= 2],
                        [0, lambda x: x, 0])

def g(x):
    return np.piecewise(x, 
                        [x < 0, (0 <= x) & (x < 2), x >= 2],
                        [0, 2, 0])
    
def mathamaticaSolution(x):
    output = np.zeros(len(x))
    for i, val in enumerate(x):
        if 0<val and val<2:
            output[i] = val**2
        elif 2<=val and val<4:
            output[i] = (4*val - val **2)
        else:
            output[i] = (0)
    return output

x = np.linspace (-20,20,1000)

mathmaticaSol = mathamaticaSolution(x)

numpySol = np.convolve(f(x),g(x),mode="same") * np.diff(x)[0]

#varConSol = vcg.varconvolve(x,f(x),g,vcg.var)

convolvedSigKernArray = convolution_2d_changing_kernel(f(x), g, x,1)

plt.plot(x,convolvedSigKernArray,linewidth=5, color = "black")
plt.plot(x,numpySol,linewidth=3,color = "blue")
plt.plot(x,mathmaticaSol, color="red")
plt.title("Plot of mathematicaSolution(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.xlim(-20,20)

plt.grid(True)
plt.show()

x = np.linspace(-20,20,1000)

def f(x):
    return np.piecewise(x, 
                        [x < 0, (0 <= x) & (x < 2), x >= 2],
                        [0, 2, 0])

def g(x):
    return np.piecewise(x, 
                        [x < 0, (0 <= x) & (x < np.pi), x >= np.pi],
                        [0, lambda x: np.sin(x), 0])

def mathematicaSol(x):
    x = np.array(x)  # Ensure input is NumPy array for vectorization
    return np.piecewise(x,
        [ 
            (np.pi < x) & (x < np.pi + 2),
            (0 < x) & (x < 2),
            (2 <= x) & (x <= np.pi)
        ],
        [
            lambda x: 2 * (1 + np.cos(2 - x)),
            lambda x: 2 * (1 - np.cos(x)),     # Simplified from -2*(-1 + cos(x))
            lambda x: -4 * np.sin(1) * np.sin(1 - x),
            0
        ]
    )

mathSol = mathematicaSol(x)

convolvedSigKernArray = convolution_2d_changing_kernel(f(x), g, x,1)

numpySol = np.convolve(f(x),g(x),mode="same") * np.diff(x)[0]

plt.plot(x, convolvedSigKernArray,linewidth=5, color= "black")
plt.plot(x, numpySol, linewidth = 3, color = "blue")
plt.plot(x, mathSol,color = "red")
plt.show()

# checking for error between all three solutions #
diffArray = []
for i in range(len(mathSol)):

    diffArray.append(mathSol[i]-convolvedSigKernArray[i])

diffArray = np.array(diffArray)

for i, array in enumerate(diffArray):
    if array != 0:
        print(array, "at", i)
