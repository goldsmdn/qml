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
    """Populate the weight matrix W using the provided x_vectors."""
    #n = W.shape[0]
    #m = x.shape[1]
    #print(f"n={n}, m={m}")
    #for i in range(n):
    #    for j in range(n):
    #        if i != j:
    #            sum_ = 0
    #            #print(f"Calculating W[{i},{j}]")
    #            for k in range(m):
    #                #print(f'i={i}, j={j} k={k}')
                   # print(f'x[i,k]={x[i,k]}, x[j,k]={x[j,k]}')
    #                sum_ += x[i, k] * x[j, k]
    #            W[i, j] = sum_ / n
     #           #print(f"W[{i},{j}] = {W[i,j]}")
    #np.fill_diagonal(W, 0)

    """
    Hebbian learning from patterns X (shape: n×m).
    W = (1/n) * X X^T with zero diagonal.
    """
    n = W.shape[0]
    # Vectorised Hebbian rule
    W[:] = (X @ X.T) / n
    # No self-connections
    np.fill_diagonal(W, 0.0)
    return W

    return W

def calculate_energy(W:np.ndarray, x:np.ndarray) -> float:
    """Calculate the energy of the system."""
    #n = W.shape[0]
    #energy = 0.0
    #for i in range(n):
    #    for j in range(n):
    #        energy += W[i, j] * x[i] * x[j]
    #return -0.5 * energy
    x_col = x.reshape(-1, 1)                   # ensure (N,1)
    val = (-0.5) * (x_col.T @ W @ x_col)       # shape (1,1)
    return float(val.item())


def calculate_update(W:np.ndarray, x:np.ndarray) -> np.ndarray:
    """Calculate the updated state vector."""
    x_new = phi(W @ x)
    return x_new

def update_asynchronous(W: np.ndarray, x: np.ndarray, order=None) -> np.ndarray:
    """One full asynchronous sweep (in-place updates)."""
    n = len(x)
    if order is None:
        order = np.arange(n)  # or np.random.permutation(n)
    x_new = x.copy()
    for i in order:
        h_i = W[i, :] @ x_new   # use already-updated states
        x_new[i] = 1 if h_i >= 0 else -1
    return x_new

def overlap(x, x_ref):
    """Calculate the overlap between two state vectors."""
    x = x.ravel() 
    x_ref = x_ref.ravel()
    return float(np.mean(x * x_ref))  # in [-1, 1]