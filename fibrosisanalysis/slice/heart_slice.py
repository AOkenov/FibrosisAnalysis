from scipy import ndimage
import numpy as np

from fibrosisanalysis.parsers import (
    ImageLoader,
    EdgeLoader
)
from fibrosisanalysis.segmentation import (
    SplineEdge,
    AngularSegments,
    RadialSegments
)


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

        Parameters
        ----------
        path : Path
            The path to the data.
        heart : str
            The name of the heart.
        slice_name : str
            The name of the slice.

        Returns
        -------
        HeartSlice
            The HeartSlice object with image and edges.
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

        self.build_slice_with_edges(image, edges)
        return self.heart_slice

    def build_from_file(self, path, heart, slice_name, n_angular, n_radial,
                        node_step=1):
        """
        Load the data for the HeartSlice object being built.
        """
        self.load_slice_data(path, heart, slice_name)
        self.build_wall_mask()
        self.build_angular_segments(n_angular, node_step)
        self.build_radial_segments(n_radial, node_step)
        self.label()
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
        self.build_wall_mask()
        self.build_angular_segments(n_angular, node_step)
        self.build_radial_segments(n_radial, node_step)
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

        return self.heart_slice

    def build_angular_segments(self, n_angular, node_step=1):
        """
        Build the angular segments.

        Parameters
        ----------
        n_angular : int
            The number of segments for angular segmentation.
        node_step : int
            The step size for radial segmentation.

        Returns
        -------
        HeartSlice
            The built HeartSlice object.
        """
        if self.heart_slice.spline_edges is None:
            raise ValueError('No spline edges detected')

        spline_edges = AngularSegments.build(self.heart_slice.spline_edges,
                                             n_angular, node_step)
        self.heart_slice.spline_edges = spline_edges
        return self.heart_slice

    def build_radial_segments(self, n_radial, node_step=1):
        """
        Build the radial segments.

        Parameters
        ----------
        n_radial : int
            The number of segments for radial segmentation.
        node_step : int
            The step size for radial segmentation.

        Returns
        -------
        HeartSlice
            The built HeartSlice object.
        """
        if self.heart_slice.spline_edges is None:
            raise ValueError('No spline edges detected')

        spline_edges = RadialSegments.build(self.heart_slice.spline_edges,
                                            n_radial, node_step)
        self.heart_slice.spline_edges = spline_edges
        return self.heart_slice

    def build_wall_mask(self):
        """
        Compute the wall mask.

        Returns
        -------
        HeartSlice
            The built HeartSlice object.
        """
        if self.heart_slice.spline_edges is None:
            raise ValueError('No spline edges detected')

        wall_mask = RadialSegments.wall_mask(self.heart_slice.image,
                                             self.heart_slice.spline_edges)
        self.heart_slice.wall_mask = wall_mask
        return self.heart_slice

    def compute_distance_map(self):
        """
        Compute the distance map.

        Returns
        -------
        HeartSlice
            The built HeartSlice object.
        """
        if self.heart_slice.wall_mask is None:
            raise ValueError('No wall mask detected')

        wall_mask = self.heart_slice.wall_mask
        labeled, num = ndimage.label(wall_mask == 0)

        if num != 2:
            raise ValueError('No endo- and epicardial layers detected')

        endo_mask = (labeled == 1) + (wall_mask == 1)
        epi_mask = (labeled == 2) + (wall_mask == 1)

        endo_dist = ndimage.distance_transform_edt(endo_mask)
        endo_dist[wall_mask == 0] = 0
        epi_dist = ndimage.distance_transform_edt(epi_mask)
        epi_dist[wall_mask == 0] = 0

        endo_epi_dist = np.zeros_like(endo_dist, dtype=float)
        np.divide(endo_dist, endo_dist + epi_dist, where=wall_mask,
                  out=endo_epi_dist)
        self.heart_slice.distance_map = endo_epi_dist
        return self.heart_slice

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
        stats['fibrosis'] = self.heart_slice.segment_fibrosis_map[
            tuple(centroids.T)
        ]
        self.heart_slice.stats = stats
        return self.heart_slice


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
    angular_segments : np.ndarray
        The labels for angular segmentation.
    radial_segments : np.ndarray
        The labels for radial segmentation.
    distance_map : np.ndarray
        The distance map.
    stats : dict
        The statistics of the fibrosis in the heart slice.
    """

    def __init__(self) -> None:
        self.image = None
        self.spline_edges = []
        self.wall_mask = None
        self.angular_segments = None
        self.radial_segments = None
        self.distance_map = None
        self.stats = None

    @property
    def angular_segments_list(self):
        """
        The labels for angular segmentation.

        Returns
        -------
        np.ndarray
            The labels for angular segmentation.
        """
        return np.arange(1, self.angular_segments.max() + 1)

    @property
    def radial_segments_list(self):
        """
        The labels for radial segmentation.

        Returns
        -------
        np.ndarray
            The labels for radial segmentation.
        """
        return np.arange(1, self.radial_segments.max() + 1)

    @property
    def total_segments_list(self):
        """
        The labels for segmentation.

        Returns
        -------
        np.ndarray
            The labels for segmentation.
        """
        return np.arange(1, self.total_segments.max() + 1)

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
        mask = (self.image == 2).astype(float)
        label_index = np.unique(labels[labels > 0])

        out = np.zeros_like(index, dtype=float)
        out[np.isin(index, label_index)] = ndimage.mean(mask, labels,
                                                        label_index)
        return out

    @property
    def segment_fibrosis_map(self):
        res = self.segment_fibrosis[self.total_segments - 1]
        res[self.wall_mask == 0] = 0
        return res