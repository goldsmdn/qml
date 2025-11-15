#quantum helper functions

def make_wires(n_qubits:int) -> list[str]:
    """Make a list of wire names based on number of qubits"""
    wires = [f'q{v+1}' for v in range (n_qubits)]
    return wires