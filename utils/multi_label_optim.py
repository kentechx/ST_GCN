import time
import trimesh
import numpy as np
import igraph

from gco import pygco


class MultiLabelOptimParams:

    def __init__(self, lambda_=10.):
        self.lambda_ = lambda_


class MultiLabelOptim:
    def __init__(self, params:MultiLabelOptimParams=None):
        self.params = params if params is not None else MultiLabelOptimParams()

    def solve(self, m: trimesh.Trimesh, probs: np.ndarray):
        """
        :param m:
        :param probs: probs for each face, (n_faces, n_class)
        :return: labels: face labels, (n_faces, )
        """
        n_class = probs.shape[1]
        edges = m.face_adjacency
        edge_weights = np.ones(edges.shape[0])
        # edge_len = np.linalg.norm(m.vertices[m.face_adjacency_edges[:, 0]]-m.vertices[m.face_adjacency_edges[:, 1]], axis=1)
        # edge_weights = edge_len
        unary_cost = np.clip(-np.log(probs), 0., 10)
        pairwise_cost = np.ones((n_class, n_class)) * self.params.lambda_
        for i in range(probs.shape[1]):
            pairwise_cost[i, i] = 0

        ls = pygco.cut_general_graph(edges, edge_weights, unary_cost, pairwise_cost, algorithm='swap')
        return ls
