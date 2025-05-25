import pytest as pt 
import numpy as np
import functions as fn 
import plotting_Double_Gauss as dg
import plotting_Saw_Square_Static as ss
import plotting_Square_SinWave as sqSin

class analytical_error_tests:
    
    def test_acc_double_gauss():
    
        set1 = dg.convolvedSigKernArray
        set2 = dg.mathmaticaSolutionArray
    
        acceptError = 0.01
    
        error = max(abs(set1 - set2))
    
        print(error)
    
        assert error < acceptError, f"Maximum error is greater than 0.01%, error of {error}"
    
    def test_saw_square():
    
        set1 = ss.convolvedSigKernArray
        set2 = ss.mathmaticaSol
    
        set1 = np.nan_to_num(set1,0.0)
        set2 = np.nan_to_num(set2,0.0)
    
        acceptError = 0.01
    
        error = max(abs(set1 - set2))
    
        assert error < acceptError, f"Maximum error is greater than 0.01%, error of {error}"
    
    def test_square_sinWave():
    
        set1 = sqSin.convolvedSigKernArray
        set2 = sqSin.mathSol
    
        set1 = np.nan_to_num(set1,0.0)
        set2 = np.nan_to_num(set2,0.0)
    
        acceptError = 0.01
    
        error = max(abs(set1 - set2))
    
        assert error < acceptError, f"Maximum error is greater than 0.01%, error of {error}"
    
    