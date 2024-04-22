import numpy as np
from skimage import measure
from fibrosisanalysis.segmentation.spline_edge import SplineEdge


class RadialSegments:
    def __init__(self):
        pass

    @staticmethod
    def build(spline_edges, number_of_segments, node_step=1):
        """Build segmentation nodes for radial segmentation.
        Each radial line is divided into a number of segments, and the nodes
        are computed for each segment.

        Parameters
        ----------
        spline_edges : list
            List of spline edges for segmentation.
        number_of_segments : int
            Number of radial segments.
        node_step : int, optional
            Step for nodes. Defaults to 1.

        Returns
        -------
            list : List of SplineEdge objects for radial segmentation.
        """
        endo_nodes = spline_edges[0].nodes
        epi_nodes = spline_edges[-1].nodes
        nodes = np.linspace(endo_nodes, epi_nodes, number_of_segments + 1)

        max_shape = np.max((spline_edges[0].full_nodes.max(axis=0),
                            spline_edges[-1].full_nodes.max(axis=0)), axis=0)
        max_shape += 1

        radial_splines = []

        for nodes_ in nodes:
            spline_edge = SplineEdge()
            spline_edge.nodes = nodes_.astype(int)
            spline_edge.ordered_nodes = nodes_.astype(int)[::node_step]
            full_nodes = spline_edge.compute(100_000)
            full_nodes = spline_edge.clip_boundaries(full_nodes, max_shape)
            spline_edge.full_nodes = full_nodes
            radial_splines.append(spline_edge)

        return radial_splines

    @staticmethod
    def label(mask, spline_edges):
        """Label radial segments based on the edges.

        Parameters
        ----------
        mask : np.ndarray
            Binary mask on which to draw edges.
        spline_edges : list
            List of SplineEdge objects.

        Returns
        -------
            np.ndarray : Labeled image with radial segments.
        """
        labels = np.zeros_like(mask, dtype='int')
        for i in range(0, len(spline_edges) - 1):
            mask_ = RadialSegments.wall_mask(mask, [spline_edges[i],
                                                    spline_edges[i+1]])
            labels[mask_] = i + 1
            labels[tuple(spline_edges[i].full_nodes.T)] = i + 1

        labels[tuple(spline_edges[-1].full_nodes.T)] = len(spline_edges) - 1
        return labels

    @staticmethod
    def wall_mask(image, spline_edges):
        """Build a mask based on the edges and input image.

        Parameters
        ----------
        image : np.ndarray
            Input image for creating the mask.
        spline_edges : dict
            Dictionary containing SplineEdge objects for mask creation.

        Returns
        -------
            np.ndarray : Binary mask created based on the edges and input image
        """
        mask = np.ones_like(image, dtype='bool')

        RadialSegments.draw_edges(mask, [spline_edges[0], spline_edges[-1]],
                                  val=0)

        labeled, num = measure.label(mask, connectivity=1, return_num=True)

        if num != 3:
            raise ValueError('Wrong number of labels {} != 3'.format(num))

        mask[labeled != 2] = 0

        return mask

    @staticmethod
    def draw_edges(mask, spline_edges, val=1):
        """Draw edges on the mask.

        Parameters
        ----------
        mask : np.ndarray
            Binary mask on which to draw edges.
        val : int, optional
            Value to set on the mask for drawing edges. Defaults to 1.
        """
        for spline_edge in spline_edges:
            coords = spline_edge.full_nodes
            mask[tuple(coords.T)] = val

        return mask
