import numpy as np
import pennylane as qml

def classical_classifier(features, weights):
    """Simple logistic: dot(features, weights), then sigmoid."""
    score = np.dot(features, weights)
    return 1 / (1 + np.exp(-score))

def encode_and_flip(features, flip_index=None):
    """Encode features into a quantum circuit, flipping one bit."""
    dev = qml.device("default.qubit", wires=len(features), shots=None)

    @qml.qnode(dev)
    def circuit():
        for i, f in enumerate(features):
            theta = 0.0 if i == flip_index and f == 1 else f * (np.pi / 2)
            qml.RY(theta, wires=i)
        return qml.probs(wires=range(len(features)))

    return circuit()

def measure_and_map_to_classical(features, flip_index=None):
    """Measure the quantum state, get a single classical bitstring."""
    probs = encode_and_flip(features, flip_index)
    measured_state = np.random.choice(len(probs), p=probs)
    bin_string = f"{measured_state:0{len(features)}b}"
    return [int(bit) for bit in bin_string]

def explain(vector, weights):
    """
    Compute feature contributions for a given vector using quantum LIME.
    
    Args:
        vector (np.array): The binary feature vector (1-D array).
        weights (np.array): Logistic regression weights (1-D array).
    
    Returns:
        np.array: Feature contributions (1-D array).
    """
    original_pred = classical_classifier(vector, weights)
    contributions = np.zeros(len(vector))

    for i, val in enumerate(vector):
        if val == 1:  # Flip only active features
            new_vec = measure_and_map_to_classical(vector, flip_index=i)
            new_pred = classical_classifier(new_vec, weights)
            contributions[i] = original_pred - new_pred

    return contributions
