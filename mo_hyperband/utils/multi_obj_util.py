import numpy as np
import sys
from loguru import logger

logger.configure(handlers=[{"sink": sys.stdout, "level": "INFO"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}

scalarization_strategy = ["random_weights", "parego", "golovin"]


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


# Adapted from Autogluon
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


# Adapted from Autogluon

def fast_nondominated_sort(values, num_samples):
    # We assume a 2d np array of dim (n_candidates, n_objectives)
    # This functions assumes a minimization problem. Implementation is based
    # on the NSGA-II paper
    assert len(values) > 0, "Need to provide at least 1 point."
    assert len(values) >= num_samples, "Need at least enough points to meet \
                                        num_samples."
    domination_counts = np.zeros(len(values))
    dominated_solutions = [[] for _ in range(len(values))]
    fronts = [[]]
    ranks = np.zeros(len(values))

    for i, v1 in enumerate(values):
        for j, v2 in enumerate(values):
            if np.alltrue(v1 < v2):  # v1 dominates v2
                dominated_solutions[i].append(j)
            elif np.alltrue(v2 < v1):  # v2 dominates v1
                domination_counts[i] += 1

        if domination_counts[i] == 0:
            ranks[i] = 1
            fronts[0].append(i)

    i = 0
    n_selected = len(fronts[0])
    while n_selected < num_samples:
        assert len(fronts[i]) > 0, "Cannot select from empty front"
        tmp = []
        for j in fronts[i]:
            for k in dominated_solutions[j]:
                domination_counts[k] -= 1
                if domination_counts[k] == 0:
                    ranks[k] = i + 1
                    tmp.append(k)
        i += 1
        n_selected += len(tmp)
        fronts.append(tmp)
    assert n_selected >= num_samples, "Could not assign enough samples"
    return ranks, fronts


def compute_eps_net(points: np.array, num_samples: int = None):
    """Sparsify a set of points returning `num_samples` points that are as
    spread out as possible. Iteratively select the point that is the furthest
    from priorly selected points.
    :param points:
    :param num_samples:
    :return: indices
    """
    assert len(points) > 0, "Need to provide at least 1 point."

    def dist(points, x):
        return np.min([np.linalg.norm(p - x) for p in points])

    n = len(points)
    eps_net = [0]
    indices_remaining = set(range(1, n))
    if num_samples is None:
        num_samples = n
    while len(eps_net) < num_samples and len(indices_remaining) > 0:
        # compute argmin dist(pts[i \not in eps_net], x)
        dist_max = -1
        best_i = 0
        for i in indices_remaining:
            cur_dist = dist(points[eps_net], points[i])
            if cur_dist > dist_max:
                best_i = i
                dist_max = cur_dist
        eps_net.append(best_i)
        indices_remaining.remove(best_i)
    return eps_net


def get_eps_net_ranking(points, num_top):
    """Produces sorted list containing the best indices
    :param points: Numpy array containing all previous evaluations
    :return: List of num_top indices
    """
    _, fronts = fast_nondominated_sort(points, num_top)
    logger.debug(f'fronts:{fronts}')
    ranked_ids = []

    i = 0
    n_selected = 0
    while n_selected < num_top:
        front = fronts[i]
        local_order = compute_eps_net(points[front],
                                      num_samples=(num_top - n_selected))
        ranked_ids += [front[j] for j in local_order]
        i += 1
        n_selected += len(local_order)
    assert len(ranked_ids) == num_top, "Did not assign correct number of \
                                        points to eps-net"
    return ranked_ids


def crowding_distance_assignment(front_points):
    assert len(front_points) > 0, "Error no empty fronts are allowed"
    distances = np.zeros(len(front_points))
    n_objectives = len(front_points[0])

    for m in range(n_objectives):  # Iterate through objectives
        vs = [(front_points[i][m], i) for i in range(len(front_points))]
        vs.sort()
        # last and first element have inf distance
        distances[vs[0][1]] = np.inf
        distances[vs[-1][1]] = np.inf
        ms = [front_points[i][m] for i in range(len(front_points))]
        scale = max(ms) - min(ms)
        if scale == 0:
            scale = 1
        for j in range(1, len(front_points) - 1):
            distances[vs[j][1]] += (vs[j + 1][0] - vs[j - 1][0]) / scale

    # determine local order
    dist_id = [(-distances[i], i) for i in range(len(front_points))]
    dist_id.sort()
    local_order = [d[1] for d in dist_id]
    return local_order


def get_nsga_ii_ranking(points, num_top):
    """Produces sorted list containing the best indices
    :param points: Numpy array containing all previous evaluations
    :return: List of num_top indices
    """
    _, fronts = fast_nondominated_sort(points, num_top)
    ranked_ids = []

    i = 0
    n_selected = 0
    while n_selected < num_top:
        front = fronts[i]
        local_order = crowding_distance_assignment(points[front])
        ranked_ids += [front[j] for j in local_order]
        i += 1
        n_selected += len(local_order)
    return ranked_ids[:num_top]


# Adapted from Autogluon
def scalarize(functional_values, mo_strategy, weights):
    """Report updated training status.

    Args:
        functional_values: Latest training result status. Reporter requires access to
        all objectives of interest.
    """
    v = np.array(functional_values)
    if mo_strategy["algorithm"] == "random_weights":
        scalarization = max([w @ v for w in weights])
    elif mo_strategy["algorithm"] == "parego":
        rho = mo_strategy.get("rho", 0.05)
        scalarization = [max(w * v) + rho * (w @ v) for w in weights]
        scalarization = max(scalarization)
    elif mo_strategy["algorithm"] == "golovin":
        scalarization = [np.min(np.clip(v / w, a_min=0, a_max=None)) ** len(w) for w in weights]
        scalarization = max(scalarization)
    else:
        raise ValueError("Specified scalarization algorithm is unknown. \
                Valid algorithms are 'random_weights' and 'parego'.")
    return scalarization


def get_scalarization_ranking(points, num_top, mo_strategy, weights):
    """Produces sorted list containing the best indices
    :param points: Numpy array containing all previous evaluations
    :return: List of num_top indices
    """
    scalarized_fitness = [scalarize(fit, mo_strategy, weights) for fit in points]
    ranked_top = np.argsort(scalarized_fitness)
    logger.debug(f'scalarized trials fitness:{scalarized_fitness}')
    logger.debug(f'sorted index:{ranked_top}')
    ranked_top = ranked_top[:num_top]
    return ranked_top
