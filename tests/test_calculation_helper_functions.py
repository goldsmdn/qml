from pathlib import Path
import numpy as np
#from pytest import approx

HOME_DIR = '..'
BASE_DIR = Path(HOME_DIR)

import sys
sys.path.append(HOME_DIR)
from src.modules.calculation_helper_functions import (phi,
                                                      generate_random_x_vector,
                                                      generate_weight_matrix,
                                                      calculate_energy,
                                                      overlap,
                                                      )

def test_phi_positive():
    """Test phi function with positive input"""
    a = 0.6
    actual_result = phi(a)
    expected_result = 1
    assert expected_result == actual_result 

def test_phi_negative():
    """Test phi function with negative input"""
    a = -0.5
    actual_result = phi(a)
    expected_result = -1.0
    assert expected_result == actual_result

def test_generate_random_x_vector():
    """Test generate_random_x_vector function"""
    n = 5
    m = 2
    x_vector = generate_random_x_vector(n, m)
    assert x_vector.shape == (n,m)
    assert np.isin(x_vector, [-1, 1]).all()

def test_generate_weight_matrix():
    """Test generate_W_matrix function"""
    n = 4
    W_matrix = generate_weight_matrix(n)
    assert W_matrix.shape == (n, n)
    assert np.all(W_matrix == 0)

def test_populate_weight_matrix():
    """Test populate_weight_matrix function"""
    n = 3
    W = generate_weight_matrix(n)
    x_vectors = np.array([[1, -1],
                          [1, 1],
                          [-1, 1]])
    from src.modules.calculation_helper_functions import populate_weight_matrix
    W_populated = populate_weight_matrix(W, x_vectors)
    expected_W = np.array([[ 0.0,         0.0, -0.66666667],
                           [ 0.0,         0.0,  0.0],
                           [-0.66666667,  0.0,  0.0]])
    assert np.allclose(W_populated, expected_W)

def test_calculate_energy():
    """Test calculate_energy function"""
    W = np.array([[0.0, 1.0],
                  [1.0, 0.0]])
    x = np.array([1, -1])
    energy = calculate_energy(W, x)
    expected_energy = 1.0
    assert energy == expected_energy

def test_overlap1():
    """Test overlap function"""
    x1 = np.array([1, -1, 1])
    x2 = np.array([1, 1, -1])
    actual_overlap = overlap(x1, x2)
    expected_overlap = -1/3
    assert actual_overlap == expected_overlap

def test_overlap2():
    """Test overlap function"""
    N = 5
    M = 1
    np.random.seed(42)
    #generate random patterns and weight matrix
    x1 = generate_random_x_vector(N, M) 
    np.random.seed(42)
    x2 = generate_random_x_vector(N, M)
    actual_overlap = overlap(x1, x2)
    expected_overlap = 1
    assert actual_overlap == expected_overlap

def test_overlap3():
    """Test overlap function"""
    N = 5
    M = 1
    np.random.seed(42)
    #generate random patterns and weight matrix
    x1 = generate_random_x_vector(N, M) 
    np.random.seed(42)
    x2 = -x1
    actual_overlap = overlap(x1, x2)
    expected_overlap = -1
    assert actual_overlap == expected_overlap