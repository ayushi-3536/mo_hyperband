from typing import List
import numpy as np
import sys
from loguru import logger

logger.configure(handlers=[{"sink": sys.stdout, "level": "INFO"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}




def pareto_index(costs: np.ndarray, index_list):
    """
    Find the pareto-optimal points
    :param costs: (n_points, m_cost_values) array
    :return: (n_points, 1) indicator if point is on pareto front or not, indices of pareto.
    """
    # first assume all points are pareto optimal
    is_pareto = np.ones(costs.shape[0], dtype=bool)

    for i, c in enumerate(costs):

        if is_pareto[i]:
            # determine all points that have a smaller cost
            all_with_lower_costs = np.any(costs < c, axis=1)
            keep_on_front = np.logical_and(all_with_lower_costs, is_pareto)
            is_pareto = keep_on_front
            is_pareto[i] = True  # keep self

    index_return = index_list[is_pareto]

    return is_pareto, index_return


def uniform_from_unit_simplex(dim):
    """Samples a point uniformly at random from the unit simplex using the
    Kraemer Algorithm. The algorithm is described here:
    https://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf

    Parameters
    ----------
    dim: int
        Dimension of the unit simplex to sample from.

    Returns:
    sample: np.array
         A point sampled uniformly from the unit simplex.
    """
    uni = np.random.uniform(size=(dim))
    uni = np.sort(uni)
    sample = np.diff(uni, prepend=0) / uni[-1]
    assert sum(sample) - 1 < 1e-6, "Error in weight sampling routine."
    return np.array(sample)

