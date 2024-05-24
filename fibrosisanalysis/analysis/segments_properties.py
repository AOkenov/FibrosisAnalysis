import numpy as np
import pandas as pd
from bitis.texture.properties import (
    DistributionEllipseBuilder
)


class SegmentsPropertiesBuilder:
    """
    Class representing structural anisotropy analysis.

    Attributes:
    ----------
    distribution_ellipses : DistributionEllipses
        Instance of DistributionEllipses class.
    """

    def __init__(self):
        self.props = pd.DataFrame()

    def build(self, heart_slice, objects_props):
        """Build segment properties.
        """
        segment_map = heart_slice.total_segments
        spline_edges = heart_slice.spline_edges

        centroids = self.compute_centroids(segment_map)
        edge_direction = self.edge_direction(spline_edges)

        props = pd.DataFrame()
        props['segment_labels'] = heart_slice.total_segments_list
        props['centroid-0'] = centroids[:, 0]
        props['centroid-1'] = centroids[:, 1]
        props['edge_direction'] = edge_direction

        anisotropy = []
        orientation = []
        width = []
        height = []

        for i in heart_slice.total_segments_list:
            props = objects_props[objects_props['segment_labels'] == i]

            dist_ellipse = DistributionEllipseBuilder().build(props)

            anisotropy.append(dist_ellipse.anisotropy)
            orientation.append(dist_ellipse.orientation)
            width.append(dist_ellipse.width)
            height.append(dist_ellipse.height)

        props['structural_anisotropy'] = np.array(anisotropy)
        props['sa_orientation'] = np.array(orientation)
        props['sa_major_axis'] = np.array(width)
        props['sa_minor_axis_'] = np.array(height)
        props['fibrosis'] = heart_slice.segment_fibrosis

        props['relative_orientation'] = self.angle_between(
            props['edge_direction'], props['sa_orientation'])

        self.props = props
        return self.props

    def angle_between(self, angle_0, angle_1):
        """
        Compute the angle between two angles.
        """
        theta = angle_0 - angle_1
        theta[theta > 0] -= ((0.5 * np.pi + theta[theta > 0]) // np.pi) * np.pi
        theta[theta < 0] += ((0.5 * np.pi - theta[theta < 0]) // np.pi) * np.pi
        return theta

    def compute_centroids(self, segment_map):
        centroids = []
        for i in range(1, segment_map.max() + 1):
            coords = np.argwhere(segment_map == i)
            centroids.append(coords.mean(axis=0))
        return np.array(centroids)

    def edge_direction(self, spline_edges):
        edge_direction = []
        for spline_edge in spline_edges[1:]:
            edge_direction.append(np.arctan2(spline_edge.direction[:, 1],
                                             spline_edge.direction[:, 0]))
        return np.concatenate(edge_direction)
