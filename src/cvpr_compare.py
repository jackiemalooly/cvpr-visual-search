import math
import numpy as np

def cvpr_compare(F1, F2):
    # Compares F1 and F2 and returns the distance between the two descriptors
    dst = np.linalg.norm(F1 - F2) # Euclidean distance
    return dst

def rank_nearest_neighbors(features: np.ndarray, query_idx: int):
    """Return sorted list of (distance, idx) for all descriptors vs. query."""
    nimg = features.shape[0]
    query_feat = features[query_idx]
    ranked_matches = []

    for candidate_idx in range(nimg):
        if candidate_idx == query_idx:
            continue
        candidate_feat = features[candidate_idx]
        distance = cvpr_compare(query_feat, candidate_feat)
        ranked_matches.append((distance, candidate_idx))

    ranked_matches.sort(key=lambda x: x[0])
    return ranked_matches

def distance_to_confidence(distance: float) -> float:
    """Convert descriptor distance to a pseudo confidence value."""
    if math.isinf(distance):
        return 0.0
    return 1.0 / (1.0 + distance)

