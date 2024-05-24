import pandas as pd
import numpy as np
from skimage import measure, morphology, segmentation

from bitis.texture.analysis import ObjectAnalysis


class ObjectsPropertiesBuilder(ObjectAnalysis):
    def __init__(self):
        super().__init__()
        self.objects_props = None

    @staticmethod
    def density(mask, intensity):
        return np.mean(intensity[mask])

    def build_from_stats(self, stats, min_area=10):
        """Selects objects with area greater than min_area and returns their
        properties in a dictionary. The properties are: orientation, centroids,
        axis_ratio, segment_labels.

        Parameters
        ----------
        stats : pd.DataFrame
            The statistics of the objects.
        min_area : int, optional
            The minimum area of the object.
        """
        stats = stats[stats['area'] >= min_area]

        major_axis = stats['major_axis_length'].to_numpy(dtype=float)
        minor_axis = stats['minor_axis_length'].to_numpy(dtype=float)
        major_axis = np.where(major_axis >= 1, 0.5 * major_axis, 0.5)
        minor_axis = np.where(minor_axis >= 1, 0.5 * minor_axis, 0.5)
        axis_ratio = major_axis / minor_axis

        self.objects_props = pd.DataFrame(stats)
        self.objects_props['major_axis_length'] = major_axis
        self.objects_props['minor_axis_length'] = minor_axis
        self.objects_props['axis_ratio'] = axis_ratio
        return self.objects_props

    def add_slice_props(self, heart_slice):
        """Adds segment id and fibrosis pecentage to objects properties.

        Parameters
        ----------
        heart_slice : HeartSlice
            The heart slice.
        """
        centroids = self.objects_props[['centroid-0',
                                        'centroid-1']].to_numpy(int)
        self.objects_props['segment_labels'] = heart_slice.total_segments[
            tuple(centroids.T)]
        self.objects_props['fibrosis'] = heart_slice.segment_fibrosis_map[
            tuple(centroids.T)]

        edge_direction = self.edge_direction(heart_slice)
        edge_direction = edge_direction[self.objects_props['segment_labels'] - 1]
        self.objects_props['edge_direction'] = edge_direction
        self.objects_props['relative_orientation'] = self.angle_between(
            self.objects_props['edge_direction'],
            self.objects_props['orientation'])

    def angle_between(self, angle_0, angle_1):
        """
        Compute the angle between two angles.
        """
        theta = angle_0 - angle_1
        theta[theta > 0] -= ((0.5 * np.pi + theta[theta > 0]) // np.pi) * np.pi
        theta[theta < 0] += ((0.5 * np.pi - theta[theta < 0]) // np.pi) * np.pi
        return theta

    def edge_direction(self, heart_slice):
        edge_direction = []
        for spline_edge in heart_slice.spline_edges[1:]:
            edge_direction.append(np.arctan2(spline_edge.direction[:, 1],
                                             spline_edge.direction[:, 0]))
        return np.concatenate(edge_direction)
