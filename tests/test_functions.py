import functions as fn 
import numpy as np

#unit tests of functions in function.py
def test_mathmaticaSolution():

    x = np.linspace(-20,20,200)

    fn.mathmatica_double_gauss(x)