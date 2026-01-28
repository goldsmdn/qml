import numpy as np

def phi(a:np.ndarray) -> np.ndarray:
    """Calculate the value of the activation function phi."""
    return np.where(a >= 0, 1, -1)

def generate_random_x_vector(n:int, m:int=1) -> np.ndarray:
    """Generate a random binary vector of size n."""
    return np.random.choice([-1, 1], size=(n,m))

def generate_weight_matrix(n:int) -> np.ndarray:
    """Generate a weight matrix W of size n x n initialized to zeros."""
    return np.zeros((n, n))

def populate_weight_matrix(W:np.ndarray, X:np.ndarray) -> np.ndarray:
    """Populate the weight matrix W using the provided x_vectors.
    Hebbian learning from patterns X (shape: n x m).
    W = (1/n) * X X^T with zero diagonal.
    """
    n = W.shape[0]
    # Vectorised Hebbian rule
    W[:] = (X @ X.T) / n
    # No self-connections
    np.fill_diagonal(W, 0.0)
    return W

def calculate_energy(W:np.ndarray, x:np.ndarray) -> float:
    """Calculate the energy of the system."""
    x_col = x.reshape(-1, 1)                   # ensure (N,1)
    val = (-0.5) * (x_col.T @ W @ x_col)       # shape (1,1)
    return float(val.item())

def calculate_update(W:np.ndarray, x:np.ndarray) -> np.ndarray:
    """Calculate the updated state vector."""
    x_updated = phi(W @ x)
    return x_updated

def update_asynchronous(W: np.ndarray, x: np.ndarray, order=None) -> np.ndarray:
    """One full asynchronous sweep (in-place updates).  Need to update one neuron at a time."""
    n = len(x)
    if order is None:
        order = np.arange(n)  #0 to n-1
    x_updated= x.copy() # make a copy to hold updated states
    for i in order:
        x_updated[i] = phi(W[i, :] @ x_updated) # just update the i-th neuron
    return x_updated

def overlap(x:np.ndarray, x_ref:np.ndarray)-> float:
    """Calculate the overlap between two state vectors."""
    x = x.ravel() 
    x_ref = x_ref.ravel()
    return float(np.mean(x * x_ref))  # in [-1, 1]
