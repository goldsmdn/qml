#test_data_helper_functions

from pathlib import Path
import numpy as np
from pytest import approx

HOME_DIR = '..'
BASE_DIR = Path(HOME_DIR)

import sys
sys.path.append(HOME_DIR)

from src.modules.data_helper_functions import (find_gamma_m, 
                                               normalise,
                                               find_norm,
                                               pre_process_feature_vector,  
                                               prepare_quantum_feature_vector,
                                               normalise_feature_vector,
                                            )                           
from config.config import ABS

def test_gamma_m1():
    """Test gamma m for a simple example"""
    x = np.array([0.5, 1])
    xm = np.array([0.3, 0.2])
    actual_result = find_gamma_m(x, xm)
    expected_result = 0.32 #hand calculated
    assert expected_result == approx(actual_result, abs=ABS)

def test_normalise():
    """Test normalise function"""
    x1 = [1, 0.6, 1.6]
    x2 = [1, 0.8, 1.2]
    actual_result = normalise(x1, x2)
    expected_result = ([0.7071067, 0.6, 0.8], ([0.7071067, 0.8, 0.6]))
    assert actual_result[0] == approx(expected_result[0], abs=1e-6)
    assert actual_result[1] == approx(expected_result[1], abs=1e-6)

def test_find_norm1():
    """Test find_norm function"""
    alpha = [0.5, 0.5, 0.5, 0.5]
    actual_result = find_norm(alpha)
    expected_result = 1.0
    assert expected_result == approx(actual_result, abs=ABS)    

def test_find_norm2():
    """Test find_norm function"""
    alpha = [1, 1, 1, 1,]
    actual_result = find_norm(alpha)
    expected_result = 2.0
    assert expected_result == approx(actual_result, abs=ABS)    

def test_pre_process_feature_vector():
    """Test pre_process_feature_vector function"""
    x1 = [0.9192568700519707,
          0.14109213343908047,
          0.8669543291655308,
          ]
    
    x2 =  [0.39365823611637335,
           0.9899964696308814,
           0.4983875912792616,
           ]
    
    y = [1, 0]
    actual_result = pre_process_feature_vector(x1, x2, y)
    expected_result = ([0.9192568700519707,
                        0.14109213343908047,
                        0.8669543291655308,
                        0.8669543291655308,
                        ],
                       [0.39365823611637335,
                        0.9899964696308814,
                        0.4983875912792616,
                        0.4983875912792616,
                        ],
                       [1, 0, 1, 0],
                       )
    assert actual_result[0] == approx(expected_result[0], abs=ABS)
    assert actual_result[1] == approx(expected_result[1], abs=ABS)
    assert actual_result[2] == expected_result[2]   

def test_prepare_quantum_feature_vector():
    """Test prepare_quantum_feature_vector function"""
    from src.modules.data_helper_functions import prepare_quantum_feature_vector
    x1 = [0.9192568700519707,
          0.14109213343908047,
          0.8669543291655308,
          0.8669543291655308,
          ]
    
    x2 =  [0.39365823611637335,
           0.9899964696308814,
           0.4983875912792616,
           0.4983875912792616,
           ]
    
    y = [1, 0, 1, 0]
    actual_result = prepare_quantum_feature_vector(x1, x2, y)
    expected_result = [0,
                       0.9192568700519707,
                       0,
                       0.39365823611637335,
                       0.14109213343908047,
                       0,
                       0.9899964696308814,
                       0,
                       0,
                       0.8669543291655308,
                       0,
                       0.4983875912792616,
                       0.8669543291655308,
                       0,
                       0.4983875912792616,
                       0,
                       ]
    assert actual_result == approx(expected_result, abs=ABS)

def test_normalise_feature_vector():
    """Test normalise_feature_vector function"""
    alpha = [0,
             0.9192568700519707,
             0,
             0.39365823611637335,
             0.14109213343908047,
             0,
             0.9899964696308814,
             0,
             0,
             0.8669543291655308,
             0,
             0.4983875912792616,
             0.8669543291655308,
             0,
             0.4983875912792616,
             0,
            ]
    actual_result = normalise_feature_vector(alpha)
    norm = np.sqrt(sum([v**2 for v in alpha]))
    expected_result = [v/norm for v in alpha]
    assert actual_result == approx(expected_result, abs=ABS)