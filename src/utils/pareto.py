import numpy as np


def nondominated_mask(objectives, directions):
    """Return which rows are nondominated across all objective columns."""
    values = np.asarray(objectives, dtype=float)
    directions = np.asarray(directions)
    if values.ndim != 2 or directions.shape != (values.shape[1],):
        raise ValueError("目标值与方向的维度不一致")
    if not np.isfinite(values).all() or not np.isin(directions, (-1, 1)).all():
        raise ValueError("目标值必须有限，目标方向必须为 1 或 -1")

    oriented = values * directions
    frontier = np.ones(len(oriented), dtype=bool)
    for index, candidate in enumerate(oriented):
        at_least_as_good = np.all(oriented >= candidate, axis=1)
        strictly_better = np.any(oriented > candidate, axis=1)
        frontier[index] = not np.any(at_least_as_good & strictly_better)
    return frontier
