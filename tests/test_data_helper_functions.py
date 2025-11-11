#test_data_helper_functions

from pathlib import Path
import numpy as np
from pytest import approx

HOME_DIR = '..'
BASE_DIR = Path(HOME_DIR)

import sys
sys.path.append(HOME_DIR)

from src.modules.data_helper_functions import find_gamma_m, normalise
from config.config import ABS

def test_gamma_m1():
    """Test gamma m for a simple example"""
    x = np.array([0.5, 1])
    xm = np.array([0.3, 0.2])
    actual_result = find_gamma_m(x, xm)
    expected_result = 0.32 #hand calculated
    assert expected_result == approx(actual_result, abs=ABS)

def test_normalise1():
    """Test normalise function"""
    x1 = [1, 0.6, 1.6]
    x2 = [1, 0.8, 1.2]
    actual_result = normalise(x1, x2)
    expected_result = ([0.7071067, 0.6, 0.8], ([0.7071067, 0.8, 0.6]))
    print(actual_result)
    print(expected_result)
    
    # Compare arrays element-wise
    assert actual_result[0] == approx(expected_result[0], abs=1e-6)
    assert actual_result[1] == approx(expected_result[1], abs=1e-6)

