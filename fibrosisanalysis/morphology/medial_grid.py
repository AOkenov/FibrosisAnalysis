import numpy as np
from skimage import (
    morphology,
    graph as sg
)
from scipy import (
    interpolate,
    spatial
)

import networkx as nx


class MedialGrid:
    def __init__(self, im, min_hole_size=100):
        self._im = im > 0

        if min_hole_size is not None:
            self._im = morphology.remove_small_holes(self._im, min_hole_size)

    def compute_medial_axis(self, im):
        skel, distance = morphology.medial_axis(im > 0, return_distance=True)

        sg_graph, nodes = sg.pixel_graph(skel, connectivity=2)
        pos = np.argwhere(skel)

        G = nx.from_scipy_sparse_array(sg_graph)

        base_cycles = nx.cycle_basis(G)

        if len(base_cycles) == 0:
            raise ValueError("No medial axis found in the image")

        medial_axis = max(base_cycles, key=len)
        # check if the medial axis is clockwise or counter-clockwise
        center = np.mean(pos[medial_axis], axis=0)
        cross_prod = np.cross(pos[medial_axis[1]] - pos[medial_axis[0]],
                              center - pos[medial_axis[0]])
        if cross_prod > 0:
            medial_axis = medial_axis[::-1]

        medial_axis_pos = pos[medial_axis]
        return medial_axis, medial_axis_pos, distance

    def compute_medial_length(self, ma_pos):
        edge_lengths = np.linalg.norm(ma_pos - np.roll(ma_pos, 1, axis=0),
                                      axis=1)
        return int(np.sum(edge_lengths))

    def compute_medial_spline(self, start_pos=None, number_of_nodes=None):
        ma, ma_pos, ma_dist = self.compute_medial_axis(self._im)

        if number_of_nodes is None:
            number_of_nodes = self.compute_medial_length(ma_pos)

        sp = SplineEdge(ma_pos)
        sp_pos = sp.compute_ordered_nodes(number_of_nodes)

        if start_pos is not None:
            sp_pos = sp.order_to_closest_node(start_pos)

        tree = spatial.KDTree(ma_pos)
        _, idx = tree.query(sp_pos)
        idx = idx.flatten()

        ma_width = ma_dist[tuple(ma_pos.T)]

        sp_width = ma_width[idx]
        return sp, sp_pos, sp_width

    def compute_radial_lines(self, sp_pos, sp_width, n_rolls=5, n_avg=11):
        radials = (np.roll(sp_pos, -n_rolls, axis=0) -
                   np.roll(sp_pos, n_rolls, axis=0))
        radials = radials / np.linalg.norm(radials, axis=1)[:, None]

        # find the radial direction
        radials = np.array([radials[:, 1], -radials[:, 0]]).T
        # smooth the radial field with a moving average
        if n_avg > 0:
            n_avg_half = n_avg // 2
            radials = np.apply_along_axis(
                lambda m: np.convolve(np.concatenate([m[-n_avg_half:], m,
                                                      m[:n_avg_half]]),
                                      np.ones(n_avg) / n_avg, mode='valid'),
                axis=0, arr=radials
            )
        radials = radials / np.linalg.norm(radials, axis=1)[:, None]
        radials = sp_width[:, None] * radials
        return radials

    def compute_grid_points(self, medial_pos, radials, n_width=20):
        ln_start = medial_pos - radials
        ln_end = medial_pos + radials

        grid_points = np.linspace(ln_start, ln_end,
                                  n_width).transpose((1, 0, 2))
        return grid_points

    def to_image_coords(self, im, grid_points):
        coords = np.argwhere(im > 0)
        medial_len = len(grid_points)
        tree = spatial.KDTree(coords)
        dist, idx = tree.query(grid_points.reshape((-1, 2)))
        idx = idx.flatten()

        grid_points = coords[idx].reshape((-1, medial_len, 2))
        return grid_points


class SplineEdge:
    def __init__(self, nodes):
        self.nodes = nodes

    def compute(self, number_of_nodes=10):
        x, y = self.nodes.T
        # Perform spline interpolation
        (t, c, k), u = interpolate.splprep([x, y], s=0., per=1)
        c = np.asarray(c).T
        bspline = interpolate.BSpline(t, c, k)

        # Evaluate the spline at specified points
        u_new = np.linspace(0, 1, number_of_nodes + 1)
        return bspline(u_new)[:-1]

    def compute_ordered_nodes(self, number_of_nodes=10):
        self.ordered_nodes = self.compute(number_of_nodes)
        return self.ordered_nodes

    def order_to_closest_node(self, node):
        node = np.atleast_2d(node)
        tree = spatial.KDTree(self.ordered_nodes)
        _, ind = tree.query(node)
        self.ordered_nodes = np.roll(self.ordered_nodes, -ind, axis=0)
        return self.ordered_nodes

    def shift_to_optimal(self, other_nodes, shift=50):

        tree = spatial.KDTree(other_nodes)
        d, _ = tree.query(self.ordered_nodes)

        res = []
        for i in range(-shift, shift):
            nodes_shifted = np.roll(self.ordered_nodes, i, axis=0)
            res.append(np.sum(np.linalg.norm(other_nodes - nodes_shifted) / d))

        optimal_shift = np.arange(-shift, shift)[np.argmin(res)]

        self.ordered_nodes = np.roll(self.ordered_nodes, optimal_shift, axis=0)

        return np.array(res), optimal_shift
