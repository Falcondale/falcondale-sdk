"""
Falcondale specific exceptions
"""


class TooManyQubitsNeeded(Exception):
    """
    Simulators and current devices cannot go beyond a given number of qubits.
    """

    def __init__(self, qubits: int, max_qubits: int):
        super().__init__(qubits, max_qubits)
        self.qubits = qubits
        self.max_qubits = max_qubits

    def __str__(self):
        return (
            f"Too many qubits required ({self.qubits} > {self.max_qubits}). "
            "Reduce the number of features using feature_selection() or preprocess(reduced_dimension=...)."
        )
