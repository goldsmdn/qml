#quantum helper functions
import pennylane as qml
import math

from src.modules.data_helper_functions import (find_norm)                              
from config.config import ABS

def make_wires(n_qubits:int) -> list[str]:
    """Make a list of wire names based on number of qubits"""
    wires = [f'q{v+1}' for v in range (n_qubits)]
    return wires

def validate_feature_list(features:list[float], wires: list[str]) -> None:
    """Validate feature list length for amplitude encoding"""
    n_qubits = len(wires)
    expected_length = 2**n_qubits
    if len(features) != expected_length:
        raise Exception(f'Feature list length {len(features)} does not match expected length {expected_length} for {n_qubits} qubits')
    
    norm = find_norm(features)
    if abs(norm - 1.0) > ABS:
        raise Exception(f'Feature vector not normalised, norm={norm:.2f}')
    
    for items in features:
        if items < 0:
            raise Exception('Feature vector contains negative values, which is not supported for amplitude encoding')

def find_theta(s:int, j:int, features:list[float]) -> float:
    """Calculate theta for amplitude encoding"""
    if s < 1 or j < 1:
        raise Exception('s and j must be >= 1')
    numerator, denominator = 0.0, 0.0

    for l in range(1, 2**(s-1) + 1):
        index = (2*j - 1) * 2**(s-1) + l - 1
        numerator += features[index]**2

    for l in range(1, 2**s + 1):
        index = (2*j - 2) * 2**(s-1) + l - 1
        denominator += features[index]**2
    if denominator == 0:
        theta = 0.0
    else:
        theta = 2 * math.asin(math.sqrt(numerator / denominator)) 
    return theta

def convert_int_to_bin_list(value:int, length:int) -> list[int]:
    """Convert integer to binary list of given length"""
    format_string = '0' + str(length) + 'b'
    bin_list = list(map(int, format(value, format_string)))
    return bin_list

def set_Paulix_controls(bin_list:list[int], wires:list[str]) -> None:
    """Set controls for Pauli-X gates based on binary list"""
    for index, item in enumerate(bin_list):
        if item == 0:
            qml.X(wires=wires[index])

def my_amplitude_encoding(features: list[float], wires: list[str]) -> None:
    """Custom amplitude encoding function"""
    validate_feature_list(features, wires)
    for s in range(len(wires), 0, -1):
        #s counts down from n_qubits to 1
        qubit = len(wires) - s + 1
        #qubit counts up from 1
        active_wire = wires[qubit-1]
        for j in range(1, 2**(qubit-1) + 1):
            theta = find_theta(s, j, features)
            if qubit == 1:
                # top wire, just apply RY
                qml.RY(theta, wires=[active_wire])
            elif qubit > 1:
                # produce a binary list and use it to set controls for the relevant wires
                bin_list = convert_int_to_bin_list(j-1, len(wires)- s)
                set_Paulix_controls(bin_list, wires)
                control_wires = wires[:qubit-1]
                # controlled RY rotation on active wire
                qml.ctrl(qml.RY, control=control_wires)(theta, wires=[active_wire])
                #undo the controls by repeating
                set_Paulix_controls(bin_list, wires)                   