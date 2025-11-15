# test quantum helper functions

from src.modules.quantum_helper_functions import make_wires

def test_make_wires():
    """Test make_wires function"""
    n_qubits = 4
    actual_result = make_wires(n_qubits)
    expected_result = ['q1', 'q2', 'q3', 'q4']
    assert actual_result == expected_result