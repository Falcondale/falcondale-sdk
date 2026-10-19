"""
Aiming for clustering dataset samples using classical, quantum-inspired and quantum techniques.
"""

from collections.abc import Callable

import numpy as np

from .data import Dataset


def prob_q_clustering(input_ds: Dataset, batch: int = 100, sigma: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Approximate Probabilistic Quantum Clustering (APQC).

    Requires the optional `clustering` dependencies: `pip install "falcondale[clustering]"`.

    Args:
        input_ds (Dataset): Preprocessed dataset
        batch (int): Batch size used by APQC
        sigma (float): (Optional) Width of the Parzen window. Estimated from the data if not provided.

    Returns:
        tuple: Label per sample and its probability
    """
    # APQC loads TensorFlow, imported lazily as it is optional (and must load after Qiskit)
    try:
        from apqc.density_estimation import DensityEstimator
        from apqc.pqc import PQC
    except ImportError as err:
        raise ImportError(
            'Quantum clustering requires optional dependencies, install them with: pip install "falcondale[clustering]"'
        ) from err

    data = input_ds.get_features().to_numpy(dtype=np.float32)
    pqc = PQC(data_gen=data, float_type=32, batch=batch, force_cpu=True)

    if sigma is None:
        estimator = DensityEstimator(data_gen=np.asarray(pqc.data_gen), batch=batch, scale=float(np.asarray(pqc.scale)))
        estimator.fit(preset_init=np.zeros((len(data), 1)))
        sigma = float(np.exp(np.asarray(estimator.log_sigma)).mean())

    pqc.set_sigmas(sigma_value=sigma)
    pqc.cluster_allocation_by_sgd()
    pqc.cluster_allocation_by_probability()

    result = pqc.basic_results[pqc.sigmas_id]
    return result["proba_labels"], result["proba_winner"]


CLUSTERING: dict[str, Callable[..., tuple[np.ndarray, np.ndarray]]] = {
    "pqc": prob_q_clustering,
}
