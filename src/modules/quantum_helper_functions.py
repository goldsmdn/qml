#quantum helper functions
import pennylane as qml
import math

from src.modules.data_helper_functions import (find_norm)                              
from config.config import ABS

def make_wires(n_qubits:int) -> list[str]:
    """Make a list of wire names based on number of qubits"""
    wires = [f'q{v+1}' for v in range (n_qubits)]
    return wires

def find_theta(s:int, j:int, features:list[float]) -> float:
    """Calculate theta for amplitude encoding"""
    if s < 1 or j < 1:
        raise Exception('s and j must be >= 1')
    numerator, denominator = 0.0, 0.0

    for l in range(1, 2**(s-1) + 1):
        print(f'{l=}')
        index = (2*j - 1) * 2**(s-1) + l - 1
        print(f'{index=}')
        numerator += features[index]**2

    for l in range(1, 2**s + 1):
        print(f'{l=}')
        index = (2*j - 2) * 2**(s-1) + l - 1
        print(f'{index=}')
        denominator += features[index]**2
    if denominator == 0:
        theta = 0.0
    else:
        theta = 2 * math.asin(math.sqrt(numerator / denominator)) 
    return theta

def convert_int_to_bin_list(value:int, length:int) -> list[int]:
    """Convert integer to binary list of given length"""
    print(f'Converting value={value} to binary list')
    #length = math.ceil(math.log2(value))
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
    norm = find_norm(features)
    if abs(norm - 1.0) > ABS:
        raise Exception(f'Feature vector not normalised, norm={norm:.2f}')
    import pennylane as qml
    #for s in range(1, len(wires)+1):
    for s in range(len(wires), 0, -1):
        qubit = len(wires) - s + 1
        print(f'Encoding on qubit {qubit} (wire {wires[qubit-1]}) with s={s}')
        active_wire = wires[qubit-1]
        for j in range(1, 2**(qubit-1) + 1):
            print(f's={s}, j={j}')
            theta = find_theta(s, j, features)
            print(f'theta={theta:.4f}')
            if qubit == 1:
                # top wire, just apply RY
                qml.RY(theta, wires=[active_wire])
            elif qubit > 1:
                print(f's={s}, j={j}')
                bin_list = convert_int_to_bin_list(j-1, len(wires)-1)
                print(f'bin_list={bin_list}')
                set_Paulix_controls(bin_list, wires)
                control_wires = wires[:qubit-1]
                print(f'creating controlled RY on wire {active_wire} with controls {control_wires}')
                qml.ctrl(qml.RY, control=control_wires)(theta, wires=[active_wire])
                set_Paulix_controls(bin_list, wires)                   