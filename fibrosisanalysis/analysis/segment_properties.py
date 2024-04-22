import numpy as np
from fibrosisanalysis.analysis import (
    DistributionEllipseBuilder,
    ObjectsPropertiesBuilder
)


class SegmentsProperties:
    """
    Class representing segment properties.

    Attributes:
    ----------
    segment_index : int
        Index of the segment.
    rx : float
        Mean x coordinate of the segment.
    ry : float
        Mean y coordinate of the segment.
    dist_ellipse : DistributionEllipse
        Instance of DistributionEllipse class.
    """

    def __init__(self):
        self.segment_index = []
        self.centroids = []
        self.objects_anisotropy = []
        self.objects_orientation = []
        self.edge_direction = []


class SegmentsPropertiesBuilder:
    """
    Class representing structural anisotropy analysis.

    Attributes:
    ----------
    distribution_ellipses : DistributionEllipses
        Instance of DistributionEllipses class.
    """

    def __init__(self):
        self.segments_properties = SegmentsProperties()

    def build(self, heart_slice):
        """Build segment properties.
        """
        # Build objects properties
        objects_props_builder = ObjectsPropertiesBuilder()
        objects_props_builder.build_from_stats(heart_slice)
        objects_props = objects_props_builder.objects_props

        segments_labels = np.arange(1, heart_slice.total_segments.max() + 1)
        self.segments_properties.segment_labels = segments_labels
        self.segments_properties.centroids = self.centroids(heart_slice)
        self.segments_properties.edge_direction = self.edge_direction(heart_slice)

        anisotropy = []
        orientation = []
        width = []
        height = []

        for i in segments_labels:
            objects_props = objects_props_builder.select_by_segment(i)

            dist_ellipse_builder = DistributionEllipseBuilder()
            dist_ellipse_builder.build(objects_props)
            dist_ellipse = dist_ellipse_builder.distribution_ellipse

            anisotropy.append(dist_ellipse.anisotropy)
            orientation.append(dist_ellipse.orientation)
            width.append(dist_ellipse.width)
            height.append(dist_ellipse.height)

        self.segments_properties.objects_anisotropy = np.array(anisotropy)
        self.segments_properties.objects_orientation = np.array(orientation)
        self.segments_properties.ellipse_width = np.array(width)
        self.segments_properties.ellipse_height = np.array(height)
        self.segments_properties.fibrosis = heart_slice.segment_fibrosis

    def centroids(self, heart_slice):
        centroids = []
        for i in range(1, heart_slice.total_segments.max() + 1):
            coords = np.argwhere(heart_slice.total_segments == i)
            centroids.append(coords.mean(axis=0))
        return np.array(centroids)

    def edge_direction(self, heart_slice):
        edge_direction = []
        for spline_edge in heart_slice.spline_edges[1:]:
            edge_direction.append(np.arctan2(spline_edge.direction[:, 1],
                                             spline_edge.direction[:, 0]))
        return np.concatenate(edge_direction)
