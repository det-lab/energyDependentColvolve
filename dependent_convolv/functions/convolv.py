import numpy as np
from inspect import signature

def convolution_2d_changing_kernel(signal, kernel, axis):
    """
    Performs a convolution of the signal and kernel with ability to perform 
    non-changing and changing kernel convolutions.
    Such as a gaussian centered at x = 2 with sigma = 4 then the function discribes a new gaussian centered at x = 4 with sigma = 8.
    Read the kernFunc(xArray, xVal) for an example of this.
    
    Parameters:
        signal (array): Any array.
        kernel (function): Any function that is used to create an array from an axis. Function can have one or two args, but do not include in args.
        axis (array): The axis array that is being used between signal and the kernel.
        

    Returns:
        array: Convolution of signal and kernel

    Example:
        >>> convolution_2d_changing_kernel([0,0,2,2,0,0], 2 * axis, [1,2,3,4,5,6])
        array
        >>> convolution_2d_chagning_kernel([0,0,2,2,0,0], 2 * axis / center, [1,2,3,4,5,6])
        array

    """

    axisDiff = np.diff(axis)

    #Ensures that the axis is uniformly spaced
    if np.allclose(axisDiff, axisDiff[0], atol=1e-8) != True:
        raise RuntimeError("Axis needs to be uniform") 

    temp = signature(kernel)
    params = temp.parameters

    argNum = len(params)
    #Ensures that the kernel function arguments are not greater than 2
    if argNum > 2:
        raise RuntimeError("Kernel function has to many arguments")

    #checks if fucntion args are 2
    if argNum == 2:
        kernelMatrix = []
        # for loop is creating a 2D matrix to be used in the convolution, the kernel remains constant but is shifted by x
        # note the second argument being used to modify the 2D matrix where each row in the matrix now has a diffrent kernel.
        for x in axis: 
            kernelMatrix.append(kernel(x-axis,x))#
        kernelMatrix = np.array(kernelMatrix)
    #checks if fucntion args are 1
    if argNum == 1:
        kernelMatrix = []
        # for loop is creating a 2D matrix to be used in the convolution, the kernel remains constant but is shifted by x
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

def mathmatica_double_gauss(x): ##### Double gaussian ######
    """
    Mathematica's integral solution to the two functions below, sigFunc() and kernFunc().
    This is an analytical solution.
    Parameters:
        x (array): Array axis

    Return:
        Array: 
    """
    numerator = 3 * np.exp(
        -1 * (np.sqrt(5) * x**2) / (2 * (9 * np.sqrt(5) + np.sqrt(5 + x**2)))
    ) * np.sqrt(2 * np.pi)

    denominator = np.sqrt(1 + (9 * np.sqrt(5)) / (np.sqrt(5 + x**2)))

    return numerator / denominator


def sigFunc(xAxis):
    """
    Creates a signal array from the axis to be used in tests.
    Parameters:
        xAxis (array): Array axis

    Return:
        Array: 
    """
    return np.exp(-1 * (xAxis**2)/18)


def kernFunc(xArray, xVal):
    """
    Creates a gaussian array with xVal being in the sigma diveation to discribe a widening gaussian as x increases
    Note that the input to xArray is an array and xVal is a float or int value.
    This is critical to having a varying convolution when used with the convolve function.
    The convolution function handles centering about an axis value and is not needed in this function.
    Parameters:
        xArray (array): Array axis.
        xVal (float): Int that discribes a modification to the function based on the centering of the function.

    Return:
        Array: 
    """
    return np.exp(-1*(xArray)**2 / (2*(np.sqrt(xVal**2/5 + 1))))


def sawWave1(x): #Saw wave for second plot
    """
    Creates a peicewise function of a saw wave.
    Parameters:
        x (array): Array axis

    Return:
        Array: 
    """
    return np.piecewise(x, 
                        [x < 0, (0 <= x) & (x < 2), x >= 2],
                        [0, lambda x: x, 0])


def squareWave1(x): #Square wave for second plot
    """
    Creates a peicewise function of a square wave.
    Parameters:
        x (array): Array axis

    Return:
        Array: 
    """
    return np.piecewise(x, 
                        [x < 0, (0 <= x) & (x < 2), x >= 2],
                        [0, 2, 0])


def mathamatica_Saw_Square(x): ##### Saw and Square wave convolution with non varying kernel #####
    """
    Creates an array of Mathematica's peicewise solution to a saw and square wave convolution.
    This function is an analytical solution.
    Parameters:
        x (array): Array axis

    Return:
        Array: 
    """
    output = np.zeros(len(x))
    for i, val in enumerate(x):
        if 0<val and val<2:
            output[i] = val**2
        elif 2<=val and val<4:
            output[i] = (4*val - val **2)
        else:
            output[i] = (0)
    return output


def squareWave2(x):
    """
    Creates a peicewise function of a square wave.
    Parameters:
        x (array): Array axis

    Return:
        Array: 
    """
    return np.piecewise(x, 
                        [x < 0, (0 <= x) & (x < 2), x >= 2],
                        [0, 2, 0])


def sinWave(x):
    """
    Creates a peicewise function of a single period sin wave.
    Parameters:
        x (array): Array axis

    Return:
        Array: 
    """
    return np.piecewise(x, 
                        [x < 0, (0 <= x) & (x < np.pi), x >= np.pi],
                        [0, lambda x: np.sin(x), 0])


def mathematica_Square_Sin(x):
    """
    Creates a peicewise function of Mathematica's solution to a square and sin wave convolution.
    This function is an analytical solution.
    Parameters:
        x (array): Array axis

    Return:
        Array: 
    """
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


### VarConvolve Github ###
#Not super optimal for what we are attempting to achive,
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