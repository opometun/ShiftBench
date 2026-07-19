"""This file contains all functions relevant for computing distances between feature vectors / distributions of feature vectors."""

from scipy.spatial.distance import jensenshannon


def js_distance(a:list, b:list, base:int=2, eps:float=1e-12) -> float:
    """
    Jensen-Shannon distance between two discrete probability distributions. 
    Bounded in [0, 1] when base=2.
    """
    a = a + eps
    b = b + eps
    a = a / a.sum()
    b = b / b.sum()
    return jensenshannon(a, b, base=base)


# TODO: Add FID