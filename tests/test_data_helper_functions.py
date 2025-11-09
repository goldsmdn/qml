#test_data_helper_functions

from pathlib import Path
import numpy as np
from pytest import approx

HOME_DIR = '..'
BASE_DIR = Path(HOME_DIR)

import sys
sys.path.append(HOME_DIR)

from src.modules.data_helper_functions import find_gamma_m
from config.config import ABS

def test_gamma_m1():
    """Test gamma m for a simple example"""
    x = np.array([0.5, 1])
    xm = np.array([0.3, 0.2])
    actual_result = find_gamma_m(x, xm)
    expected_result = 0.32 #hand calculated
    assert expected_result == approx(actual_result, abs=ABS)