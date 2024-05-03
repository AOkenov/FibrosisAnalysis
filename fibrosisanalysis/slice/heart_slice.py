from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from fibrosisanalysis.parsers import (
    ImageLoader,
    EdgeLoader
)
from fibrosisanalysis.segmentation import (
    SplineEdge,
    AngularSegments,
    RadialSegments
)


class HeartSlice:
    """
    Represents a slice of the heart.

    Attributes
    ----------
    image : np.ndarray
        The image of the heart slice.
    spline_edges : list
        A list containing spline edges.
    wall_mask : np.ndarray
        The mask representing the heart wall.
    angular_labels : np.ndarray
        The labels for angular segmentation.
    radial_labels : np.ndarray
        The labels for radial segmentation.
    """

    def __init__(self) -> None:
        self.image = None
        self.spline_edges = []
        self.wall_mask = None
        self.angular_segments = None
        self.radial_segments = None
        self.stats = None

    @property
    def total_segments(self):
        """
        The labels for segmentation.

        Returns
        -------
        np.ndarray
            The labels for segmentation.
        """
        segments = ((self.radial_segments - 1) * self.angular_segments.max()
                    + self.angular_segments)

        segments[self.wall_mask == 0] = 0
        return segments

    @property
    def pixel_per_segment(self):
        """
        The number of pixels per segment.

        Returns
        -------
        np.ndarray
            The number of pixels per segment.
        """
        return np.bincount(self.total_segments.ravel())[1:]

    @property
    def segment_fibrosis(self):
        """
        The density for segmentation.

        Returns
        -------
        np.ndarray
            The density for segmentation.
        """
        index = np.arange(1, self.total_segments.max() + 1)
        labels = self.total_segments.copy()
        labels[self.image == 0] = 0
        return ndimage.mean(self.image, labels, index) - 1

    @property
    def segment_fibrosis_map(self):
        res = self.segment_fibrosis[self.total_segments - 1]
        res[self.wall_mask == 0] = 0
        return res


class HeartSliceBuilder:
    """
    A class that builds a HeartSlice object.

    Attributes
    ----------
    heart_slice : HeartSlice
        The HeartSlice object being built.
    """

    def __init__(self):
        self.heart_slice = HeartSlice()

    def load_slice_data(self, path, heart, slice_name):
        """
        Load the data for the HeartSlice object being built.
        """
        image_loader = ImageLoader(path)
        image = image_loader.load_slice_data(path.joinpath(heart, 'Images',
                                                           slice_name))

        edge_loader = EdgeLoader(path)

        edges = {}
        for i, name in enumerate(['epi', 'endo']):
            filename = '{}_{}'.format(slice_name, i)
            nodes = edge_loader.load_slice_data(path.joinpath(heart, 'Edges',
                                                              filename))
            edges[name] = nodes[:, [1, 0]]

        return image, edges

    def build_from_file(self, path, heart, slice_name, n_angular, n_radial,
                        node_step=1):
        """
        Load the data for the HeartSlice object being built.
        """
        image, edges = self.load_slice_data(path, heart, slice_name)
        self.build(image, edges, n_angular, n_radial, node_step)
        return self.heart_slice

    def build(self, image, edges, n_angular, n_radial, node_step=1):
        """
        Build a HeartSlice object.

        Parameters
        ----------
        image : np.ndarray
            The image of the heart slice.
        edges : dict
            The nodes representing the endo- and epicardial spline edge.
        n_angular : int
            The number of segments for angular segmentation.
        n_radial : int
            The number of segments for radial segmentation.
        """
        self.build_slice_with_edges(image, edges)
        self.build_segments(n_angular, n_radial, node_step)
        self.label()
        return self.heart_slice

    def build_slice_with_edges(self, image, edges):
        self.heart_slice = HeartSlice()
        self.heart_slice.image = image
        self.add_spline_edge(edges['endo'], image.shape)
        self.add_spline_edge(edges['epi'], image.shape)
        return self.heart_slice

    def add_spline_edge(self, nodes, shape):
        """
        Build a spline edge.

        Parameters
        ----------
        nodes : np.ndarray
            The nodes representing the spline edge.
        shape : tuple 
            The shape of the image.

        Returns
        -------
        SplineEdges
            The built SplineEdge object.
        """
        spline_edge = SplineEdge()
        spline_edge.nodes = nodes

        full_nodes = spline_edge.compute(100_000)
        full_nodes = spline_edge.clip_boundaries(full_nodes, shape)

        if not spline_edge.check_connectivity(full_nodes, shape):
            raise ValueError('Not enough points to draw edge')

        spline_edge.full_nodes = full_nodes
        self.heart_slice.spline_edges.append(spline_edge)

        return spline_edge

    def build_segments(self, n_angular, n_radial, node_step=1):
        """
        Build the segments.

        Parameters
        ----------
        n_angular_segments : int
            The number of segments for angular segmentation.
        n_radial_segments : int
            The number of segments for radial segmentation.
        node_step : int
            The step size for radial segmentation.

        Returns
        -------
        HeartSlice
            The built HeartSlice object.
        """
        spline_edges = AngularSegments.build(self.heart_slice.spline_edges,
                                             n_angular, node_step)
        spline_edges = RadialSegments.build(spline_edges, n_radial, node_step)

        wall_mask = RadialSegments.wall_mask(self.heart_slice.image,
                                             spline_edges)
        self.heart_slice.wall_mask = wall_mask

        # import matplotlib.pyplot as plt

        # plt.figure()
        # plt.imshow(self.heart_slice.image)

        # for spline_edge in spline_edges:
        #     plt.plot(spline_edge.full_nodes[:, 1],
        #              spline_edge.full_nodes[:, 0],
        #              'k', lw=0.5)
        # plt.show()

        self.heart_slice.spline_edges = spline_edges

    def label(self):
        """
        Label the segments.

        Returns
        -------
        HeartSlice
            The built HeartSlice object.
        """
        wall_mask = self.heart_slice.wall_mask
        spline_edges = self.heart_slice.spline_edges
        self.heart_slice.angular_segments = AngularSegments.label(wall_mask,
                                                                  spline_edges)
        self.heart_slice.radial_segments = RadialSegments.label(wall_mask,
                                                                spline_edges)
        return self.heart_slice

    def add_stats(self, stats):
        centroids = stats[['centroid-0', 'centroid-1']].to_numpy(dtype=int)
        stats['segment'] = self.heart_slice.total_segments[tuple(centroids.T)]
        stats['fibrosis'] = self.heart_slice.segment_fibrosis_map[tuple(centroids.T)]
        self.heart_slice.stats = stats
        return self.heart_slice
