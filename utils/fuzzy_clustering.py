import time

import numpy as np
import trimesh
import igraph

from sklearn.cluster import KMeans


class FuzzyClusteringParams:
    # fuzzy_area_per = [0.,
    #                   0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,     # 11-18
    #                   0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,     # 21-28
    #                   0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,     # 31-38
    #                   0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]     # 41-48

    def __init__(self, fuzzy_area_per=0.5):
        self.fuzzy_area_per = [fuzzy_area_per for _ in range(33)]


class FuzzyClustering:
    """
    :param yita: angular distance ::= yita * (1- cos(a)) if the interior angle < 180;
                                      1-cos(a), else.
    """

    def __init__(self, yita=0.05, params: FuzzyClusteringParams=None):
        self.yita = yita
        self.params = params if params is not None else FuzzyClusteringParams()

    def solve_all(self, m: trimesh.Trimesh, ls: np.ndarray) -> np.ndarray:
        """
        :param m:
        :param ls: face labels with range [0, 17]
        :return: out_ls: output face labels
        """
        g = igraph.Graph(n=m.faces.shape[0], edges=m.face_adjacency)

        for l in np.unique(ls):
            if l > 0:
                ls = self.solve_one(m, g, ls, l)

        return ls

    def solve_one(self, m: trimesh.Trimesh, g: igraph.Graph, f_labels: np.ndarray, label: int) -> np.ndarray:
        """
        :param m:
        :param g: the graph corresponding to mesh `m`
        :param f_labels: face labels of mesh `m`
        :param label: tooth label
        :return out_f_labels: output face labels
        """
        cur_f_idx = np.where(f_labels == label)[0]
        fuzzy_f_idx = self.get_fuzzy_region(m, g, f_labels, cur_f_idx, self.params.fuzzy_area_per[label])

        # 1. create a graph for the fuzzy region
        fuzzy_m = trimesh.Trimesh(m.vertices, m.faces[fuzzy_f_idx])
        fuzzy_ls = f_labels[fuzzy_f_idx]
        cap = self.calc_capacity(fuzzy_m)
        fuzzy_g = igraph.Graph(n=fuzzy_m.faces.shape[0], edges=fuzzy_m.face_adjacency, edge_attrs={'cap':cap})

        # 2. add terminal nodes
        fuzzy_boundary_f_idx = np.where(np.array(fuzzy_g.degree())<3)[0]
        tooth_f_idx = fuzzy_boundary_f_idx[fuzzy_ls[fuzzy_boundary_f_idx]==label]
        gum_f_idx = np.setdiff1d(fuzzy_boundary_f_idx, tooth_f_idx)

        tooth_node = max(fuzzy_g.vs.indices) + 1
        gum_node = tooth_node+1
        fuzzy_g.add_vertex(tooth_node)
        fuzzy_g.add_vertex(gum_node)
        fuzzy_g.add_edges([[tooth_node, i] for i in tooth_f_idx], {'cap': [1000000 for _ in range(len(tooth_f_idx))]})
        fuzzy_g.add_edges([[gum_node, i] for i in gum_f_idx], {'cap': [1000000 for _ in range(len(gum_f_idx))]})

        # 3. cut
        ret = fuzzy_g.mincut(tooth_node, gum_node, 'cap')
        out_tooth_idx = list(set(ret.partition[0]) - {tooth_node})
        out_gum_idx = list(set(ret.partition[1]) - {gum_node})

        out_tooth_idx = fuzzy_f_idx[out_tooth_idx]
        out_gum_idx = fuzzy_f_idx[out_gum_idx]

        # 4. update labels
        out_f_labels = f_labels.copy()
        out_f_labels[out_tooth_idx] = label
        idx = np.where(out_f_labels[out_gum_idx]==label)[0]
        out_f_labels[np.array(out_gum_idx)[idx]] = 0

        return out_f_labels

    def calc_projection(self, m: trimesh.Trimesh) -> np.ndarray:
        c = m.triangles_center[m.face_adjacency]
        c01 = (c[:, 1]-c[:, 0])
        c01 /= np.linalg.norm(c01, axis=1, keepdims=True)
        proj = np.sum(c01 * m.face_normals[m.face_adjacency[:, 0]], 1)
        # is_convex = signs < 0.0        # dynamic threshold?
        return proj

    def calc_capacity(self, m: trimesh.Trimesh):
        # calc convex
        proj = self.calc_projection(m)
        # kmeans = KMeans(2, random_state=0).fit(proj.reshape((-1, 1)))
        # proj_thresh = min(proj[np.where(kmeans.labels_==1)].max(), proj[np.where(kmeans.labels_==0)].max())
        # is_convex = proj < max(proj_thresh, 0.)       ##
        is_convex = proj < 0.
        edge_len = np.linalg.norm(m.vertices[m.face_adjacency_edges[:, 0]]-m.vertices[m.face_adjacency_edges[:, 1]], axis=1)

        # assign yita
        yita = np.ones((m.face_adjacency.shape[0]))
        yita[is_convex] = 0.05

        # calculate capacity
        angular_dists = yita * (1 - np.cos(m.face_adjacency_angles))
        # cap = 1. / (1 + angular_dists/ angular_dists.mean())
        cap = np.exp(-angular_dists / angular_dists.mean())
        cap *= edge_len

        # face_colors = np.zeros((len(m.faces), 3), dtype=int)
        # for x in m.face_adjacency:
        #     i, j = x[0], x[1]
        #     face_colors[x[0]][0] = min(255, max(int(angular_dists[x[0]] *20 * 255), face_colors[x[0]][0]))
        #     face_colors[x[1]][0] = min(255, max(int(angular_dists[x[1]] *20 * 255), face_colors[x[1]][0]))
        # trimesh.Trimesh(m.vertices, m.faces, face_colors=face_colors).show()

        return cap

    def get_fuzzy_region(self, m:trimesh.Trimesh, g: igraph.Graph, f_labels: np.ndarray, f_idx: np.ndarray, area_per: float) -> np.ndarray:
        """
        Get the fuzzy region by dilating on the boundary faces.
        :param m:
        :param g: the face graph corresponding to mesh `m`
        :param f_labels: face labels
        :param f_idx: the selected face indices
        :param area_per:  area percentage of dilated faces
        :return: face indices of the fuzzy region
        """
        cur_label = f_labels[f_idx[0]]
        init_area = m.area_faces[f_idx].sum()
        new_area = 0.

        subg = g.subgraph(vertices=f_idx)
        cur_f_idx = f_idx[np.array(subg.degree())<3]        # boundary
        while new_area / init_area < area_per:
            subg = g.subgraph(vertices=cur_f_idx)
            boundary_f_idx = cur_f_idx[np.array(subg.degree())<3]
            idx = g.neighborhood(vertices=boundary_f_idx, order=1)
            # update
            cur_f_idx = np.unique(np.concatenate([cur_f_idx, np.unique([i for l in idx for i in l])]))
            cur_f_idx = cur_f_idx[np.logical_or(f_labels[cur_f_idx]==0, f_labels[cur_f_idx]==cur_label)]
            new_area = m.area_faces[cur_f_idx].sum()

        return cur_f_idx

    def dilate(self, m: trimesh.Trimesh, g: igraph.Graph, f_idx: np.ndarray, area_per: float) -> np.ndarray:
        """
        :param m:
        :param g: the face graph corresponding to mesh `m`
        :param f_idx: the selected face indices
        :param area_per:  area percentage of dilated faces
        :return: dilated face indices, including the `f_idx`
        """
        init_area = m.area_faces[f_idx].sum()
        new_area = 0.

        new_f_idx = f_idx.copy()
        while new_area / init_area < area_per:
            subg = g.subgraph(vertices=new_f_idx)
            boundary_f_idx = new_f_idx[np.array(subg.degree())<3]
            idx = g.neighborhood(vertices=boundary_f_idx, order=1)
            new_idx = np.setdiff1d(np.unique(idx), new_f_idx)
            new_area += m.area_faces[new_idx].sum()
            # update
            new_f_idx = np.unique(np.concatenate([new_f_idx, new_idx]))

        return new_f_idx

    def _show(self, m: trimesh.Trimesh, f_idx: np.ndarray):
        face_colors = np.ones((m.faces.shape[0], 3), dtype=int) * 170
        face_colors[f_idx] = [170, 170, 0]
        trimesh.Trimesh(m.vertices, m.faces, face_colors=face_colors).show()

