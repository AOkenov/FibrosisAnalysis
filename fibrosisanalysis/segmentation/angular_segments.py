import numpy as np
from skimage import morphology
from scipy import ndimage
from fibrosisanalysis.segmentation.distance import Distance
from fibrosisanalysis.segmentation.radial_lines import RadialLines


class AngularSegments:
    def __init__(self):
        pass

    @staticmethod
    def label(mask, spline_edges):
        """Label angular segments based on the edges.

        Parameters
        ----------
        mask : np.ndarray
            Binary mask on which to draw edges.
        spline_edges : list
            List of SplineEdge objects.

        Returns
        -------
            np.ndarray : Labeled image with angular segments.

        Raises
        ------
        ValueError
            If the endo and epi nodes have different lengths.

        Notes
        -----
        Individual segments should be smaller that 1/2 of the mask size.
        """
        endo_nodes = spline_edges[0].ordered_nodes
        epi_nodes = spline_edges[-1].ordered_nodes

        if len(endo_nodes) != len(epi_nodes):
            raise ValueError('Endo and epi nodes have different lengths.')

        if len(endo_nodes) < 3:
            return mask.astype('int')

        labels = np.zeros_like(mask, dtype='int')

        for i in range(0, len(endo_nodes)):
            mask_ = mask.astype('int')
            line_0 = RadialLines.draw(endo_nodes[i], epi_nodes[i])
            mask_[tuple(line_0.T)] = 0

            i_1 = (i + 1) % len(endo_nodes)
            line_1 = RadialLines.draw(endo_nodes[i_1], epi_nodes[i_1])
            mask_[tuple(line_1.T)] = 0

            labels_, num = morphology.label(
                mask_, connectivity=1, return_num=True)

            areas = ndimage.sum_labels(mask_, labels_,
                                       index=np.arange(1, num + 1))
            area_max_label = np.argmax(areas) + 1

            labels[(labels_ != area_max_label) & (mask > 0)] = i + 1
            labels[tuple(line_0.T)] = i + 1

            if len(endo_nodes) == 3:
                labels[labels_ == area_max_label] = i + 2
                labels[tuple(line_1.T)] = i + 2
                return labels

        return labels

    @staticmethod
    def build(spline_edges, number_of_segments, node_step=3,
              optimization_step=5):
        """Compute the starting and ending points of radial lines for
        segmentation.

        Parameters
        ----------
        spline_edges : list
            List of SplineEdge objects.
        number_of_segments : int
            Number of angular segments.
        node_step : int
            Step size for angular segmentation.
        optimization_step : int
            Step size for optimization of radial lines.

        Returns
        -------
        tuple
            Tuple containing starting and ending points of radial lines.
        """

        number_of_segments = number_of_segments * node_step

        if number_of_segments < 36:
            raise ValueError('Number of segments should be at least 36.')

        epi_nodes = spline_edges[-1].sample_nodes(number_of_segments)

        number_of_nodes = optimization_step * number_of_segments
        potential_endo_nodes = spline_edges[0].sample_nodes(number_of_nodes)
        _, ind = Distance.shortest(potential_endo_nodes, epi_nodes[0])

        potential_endo_nodes = np.roll(potential_endo_nodes, -ind, axis=0)

        endo_nodes = AngularSegments._optimize(epi_nodes,
                                               potential_endo_nodes,
                                               optimization_step)

        endo_nodes = Distance.closest_neighbour(spline_edges[0].full_nodes,
                                                endo_nodes)
        epi_nodes = Distance.closest_neighbour(spline_edges[-1].full_nodes,
                                               epi_nodes)
        spline_edges[0].nodes = endo_nodes
        spline_edges[-1].nodes = epi_nodes
        spline_edges[0].ordered_nodes = endo_nodes[::node_step]
        spline_edges[1].ordered_nodes = epi_nodes[::node_step]

        return spline_edges

    @staticmethod
    def _optimize(epi_nodes, endo_nodes, optimization_step, search_range=None):
        """Optimize the radial lines based on the shortest distance
        between them. The method look for combination of `endo_nodes` which
        minimize normed distance between pairs of endo and epi nodes.

        Parameters
        ----------
        epi_nodes : np.ndarray
            Starting points of radial lines.
        endo_nodes : np.ndarray
            Ending points of radial lines.
        search_range : None, optional
            Range of endo nodes indexes for minimization of cost of alignment
            The method will use range `[-search_range, search_range)`.
            If None, `search_range = len(endo_nodes) // 4`.

        Returns
        -------
        np.ndarray
            Optimized radial lines.
        """
        shortest_distance, _ = Distance.shortest(endo_nodes, epi_nodes)

        if search_range is None:
            search_range = len(endo_nodes) // 4

        cost = []
        for i in range(-search_range, search_range):
            endo_nodes_ = np.roll(endo_nodes, i,
                                  axis=0)[::optimization_step]
            cost.append(AngularSegments._cost(epi_nodes, endo_nodes_,
                                              shortest_distance))

        ind = np.argmin(cost)

        shift = np.arange(-search_range, search_range)

        return np.roll(endo_nodes, shift[ind], axis=0)[::optimization_step]

    def _cost(epi_nodes, endo_nodes, shortest_distance):
        """Calculate the cost of aligning radial lines.

        Parameters
        ----------
        epi_nodes : np.ndarray
            Starting points of radial lines.
        endo_nodes : np.ndarray
            Ending points of radial lines.
        shortest_distance : float
            Shortest distance between radial lines.

        Returns
        -------
        float
            Cost of alignment.
        """
        distance = Distance.between(epi_nodes, endo_nodes)
        cost = np.sum(((distance - shortest_distance)
                       / shortest_distance) ** 2)

        return cost
