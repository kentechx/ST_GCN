import numpy as np

def calc_areas(vertices, centers):
    # vertices: (n, 9)
    v = vertices + np.tile(centers, (1, 3))
    normals = np.cross(v[:, 3:6] - v[:, :3],
                       v[:, 6:] - v[:, :3])
    areas = 0.5 * np.sqrt((normals ** 2).sum(axis=1))

    return areas

def weighted_iou(pred, y, weights):
    inter = np.sum(((y > 0) & (y == pred)) * weights)
    union = np.sum(((y > 0) | (pred > 0)) * weights)
    return inter / union

def weighted_acc(pred, y, weights):
    inter = np.sum((y == pred) * weights)
    return inter / weights.sum()
