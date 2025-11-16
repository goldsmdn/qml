# test quantum helper functions

import numpy as np
import math
import pennylane as qml

from src.modules.quantum_helper_functions import make_wires
from src.modules.quantum_helper_functions import (my_amplitude_encoding, 
                                                  find_theta,
                                                  convert_int_to_bin_list,
                                                  )

from config.config import ABS

def test_make_wires():
    """Test make_wires function"""
    n_qubits = 4
    actual_result = make_wires(n_qubits)
    expected_result = ['q1', 'q2', 'q3', 'q4']
    assert actual_result == expected_result

def test_my_amplitude_encoding1():
    """Test my_amplitude_encoding function"""
    features = [0.0, 1.0]
    wires = ['q1']
    
    dev = qml.device('default.qubit', wires=wires)
    
    @qml.qnode(dev)
    def circuit():
        my_amplitude_encoding(features, wires)
        return qml.state()
    
    actual_state = circuit()
    expected_state = np.array(features)
    assert np.allclose(actual_state, expected_state)

def test_find_theta11():
    """Test find_theta function"""
    features =  [0 for i in range(8)]
    features[0] = math.sqrt(0.2)
    features[2] = math.sqrt(0.5)
    features[6] = math.sqrt(0.2)
    features[7] = math.sqrt(0.1)
    s = 1
    j = 1
    actual_theta = find_theta(s, j, features)
    expected_theta = 0
    assert np.isclose(actual_theta, expected_theta) 

def test_find_theta12():
    """Test find_theta function"""
    features =  [0 for i in range(8)]
    features[0] = math.sqrt(0.2)
    features[2] = math.sqrt(0.5)
    features[6] = math.sqrt(0.2)
    features[7] = math.sqrt(0.1)
    s = 1
    j = 2
    actual_theta = find_theta(s, j, features)
    expected_theta = 0
    assert np.isclose(actual_theta, expected_theta)

def test_find_theta14():
    """Test find_theta function"""
    features =  [0 for i in range(8)]
    features[0] = math.sqrt(0.2)
    features[2] = math.sqrt(0.5)
    features[6] = math.sqrt(0.2)
    features[7] = math.sqrt(0.1)
    s = 1
    j = 4
    actual_theta = find_theta(s, j, features)
    expected_theta = 1.231
    assert np.isclose(actual_theta, expected_theta, atol=ABS)  

def test_find_theta21():
    """Test find_theta function"""
    features =  [0 for i in range(8)]
    features[0] = math.sqrt(0.2)
    features[2] = math.sqrt(0.5)
    features[6] = math.sqrt(0.2)
    features[7] = math.sqrt(0.1)
    s = 2
    j = 1
    actual_theta = find_theta(s, j, features)
    expected_theta = 2.014
    assert np.isclose(actual_theta, expected_theta, atol=ABS)  

def test_find_theta22():
    """Test find_theta function"""
    features =  [0 for i in range(8)]
    features[0] = math.sqrt(0.2)
    features[2] = math.sqrt(0.5)
    features[6] = math.sqrt(0.2)
    features[7] = math.sqrt(0.1)
    s = 2
    j = 2
    actual_theta = find_theta(s, j, features)
    expected_theta = 3.142
    assert np.isclose(actual_theta, expected_theta, atol=ABS)  

def test_find_theta31():
    """Test find_theta function"""
    features =  [0 for i in range(8)]
    features[0] = math.sqrt(0.2)
    features[2] = math.sqrt(0.5)
    features[6] = math.sqrt(0.2)
    features[7] = math.sqrt(0.1)
    s = 3
    j = 1
    actual_theta = find_theta(s, j, features)
    expected_theta = 1.159
    assert np.isclose(actual_theta, expected_theta, atol=ABS)  

def test_my_amplitude_encoding2():
    """Test my_amplitude_encoding function"""
    features = [1.0, 0.0]
    wires = ['q1']
    
    dev = qml.device('default.qubit', wires=wires)
    
    @qml.qnode(dev)
    def circuit():
        my_amplitude_encoding(features, wires)
        return qml.state()
    
    actual_state = circuit()
    expected_state = np.array(features)
    assert np.allclose(actual_state, expected_state)

def test_my_amplitude_encoding3():
    """Test my_amplitude_encoding function"""
    features = [0.6, 0.8]
    wires = ['q1']
    
    dev = qml.device('default.qubit', wires=wires)
    
    @qml.qnode(dev)
    def circuit():
        my_amplitude_encoding(features, wires)
        return qml.state()
    
    actual_state = circuit()
    expected_state = np.array(features)
    assert np.allclose(actual_state, expected_state)

def test_my_amplitude_encoding_4():
    """Test my_amplitude_encoding function"""
    features = [0.5, 0.5, 0.5, 0.5]
    wires = ['q1', 'q2']
    
    dev = qml.device('default.qubit', wires=wires)
    
    @qml.qnode(dev)
    def circuit():
        my_amplitude_encoding(features, wires)
        return qml.state()
    
    actual_state = circuit()
    expected_state = np.array(features)
    assert np.allclose(actual_state, expected_state)

def test_my_amplitude_encoding_4():
    """Test my_amplitude_encoding function"""
    features = [0.6, 0.8, 0.0, 0.0]
    wires = ['q1', 'q2']
    
    dev = qml.device('default.qubit', wires=wires)
    
    @qml.qnode(dev)
    def circuit():
        my_amplitude_encoding(features, wires)
        return qml.state()
    
    actual_state = circuit()
    expected_state = np.array(features)
    assert np.allclose(actual_state, expected_state)

def test_my_amplitude_encoding_4():
    """Test my_amplitude_encoding function"""
    features = [0.6, 0.0, 0.0, 0.8]
    wires = ['q1', 'q2']
    
    dev = qml.device('default.qubit', wires=wires)
    
    @qml.qnode(dev)
    def circuit():
        my_amplitude_encoding(features, wires)
        return qml.state()
    
    actual_state = circuit()
    expected_state = np.array(features)
    assert np.allclose(actual_state, expected_state)

def test_my_amplitude_encoding_not_normalised():
    """Test my_amplitude_encoding function raises exception for non-normalised vector"""
    features = [1.0, 1.0]
    wires = ['q1']
    
    dev = qml.device('default.qubit', wires=wires)
    
    @qml.qnode(dev)
    def circuit():
        my_amplitude_encoding(features, wires)
        return qml.state()
    
    try:
        circuit()
        assert False, "Expected exception for non-normalised vector"
    except Exception as e:
        assert str(e) == 'Feature vector not normalised, norm=1.41'

def test_convert_int_to_bin_list_6():
    """Test convert_int_to_bin_list function"""
    value = 6
    length = 3
    actual_result = convert_int_to_bin_list(value, length)
    expected_result = [1, 1, 0]
    assert actual_result == expected_result

def test_convert_int_to_bin_list_7():
    """Test convert_int_to_bin_list function"""
    value = 7
    length = 4
    actual_result = convert_int_to_bin_list(value, length)
    expected_result = [0, 1, 1, 1]
    assert actual_result == expected_result

def test_convert_int_to_bin_list_8():
    """Test convert_int_to_bin_list function"""
    length = 4
    value = 8
    actual_result = convert_int_to_bin_list(value, length)
    expected_result = [1, 0, 0, 0]
    assert actual_result == expected_result

def test_convert_int_to_bin_list_0():
    """Test convert_int_to_bin_list function"""
    length = 4
    value = 0
    actual_result = convert_int_to_bin_list(value, length)
    expected_result = [0, 0, 0, 0]
    assert actual_result == expected_result