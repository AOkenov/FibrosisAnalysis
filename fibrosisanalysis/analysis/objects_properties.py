import pandas as pd
import numpy as np
from skimage import measure, morphology, segmentation

from fibrosisanalysis.segmentation.distance import Distance


class ObjectsPropertiesBuilder:
    def __init__(self):
        self.objects_props = None

    def build_from_segment(self, image, min_area=10):
        """Builds objects properties from a segment.

        Parameters
        ----------
        image : np.ndarray
            The image of the segment.
        min_area : int, optional
            The minimum area of the object.
        """
        mask = morphology.remove_small_objects(image > 0, min_area)
        mask = segmentation.clear_border(mask)
        labels = measure.label(mask, connectivity=1)
        props = measure.regionprops_table(labels,
                                          properties=('area',
                                                      'centroid',
                                                      'major_axis_length',
                                                      'minor_axis_length',
                                                      'orientation'))
        major_axis = props['major_axis_length']
        minor_axis = props['minor_axis_length']
        major_axis = np.where(major_axis >= 1, 0.5 * major_axis, 0.5)
        minor_axis = np.where(minor_axis >= 1, 0.5 * minor_axis, 0.5)
        # # Swap axes due matplotlib plotting
        # orientation = 0.5 * np.pi - props['orientation']

        self.objects_props = pd.DataFrame()

        for k, v in props.items():
            self.objects_props[k] = v

        # self.objects_props['orientation'] = orientation
        self.objects_props['major_axis_length'] = major_axis
        self.objects_props['minor_axis_length'] = minor_axis
        self.objects_props['axis_ratio'] = major_axis / minor_axis
        self.objects_props['segment_labels'] = 1
        return self.objects_props

    @staticmethod
    def compute(labels, density_map, props_list, extra_props=[]):
        props = measure.regionprops_table(labels, intensity_image=density_map,
                                          properties=props_list,
                                          extra_properties=extra_props)

        return pd.DataFrame(props)
        # if self.props is None:
        #     self.props = props
        #     return
        #
        # self.props = self.props.merge(props, on='label', suffixes=('_old', ''))
        # drop_list = [
        #     column for column in self.props.columns if '_old' in column]
        # self.props.drop(columns=drop_list, inplace=True)

    @staticmethod
    def distance(coords, edges):
        d_endo, _ = Distance.shortest(edges['endo'].compute(), coords)
        d_epi, _ = Distance.shortest(edges['epi'].compute(), coords)
        return d_endo / (d_endo + d_epi)

    @staticmethod
    def tangent(coords, edge_props):
        edge_coords = edge_props[['centroid-0', 'centroid-1']].to_numpy()
        _, inds = Distance.shortest(edge_coords, coords)
        return edge_props['orientation'][inds].to_numpy()

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
        orientation = stats['orientation'].to_numpy(dtype=float)
        # orientation = np.where(orientation < 0, orientation + np.pi,
        #                        orientation)

        # Swap axes due matplotlib plotting
        # orientation = 0.5 * np.pi - orientation

        self.objects_props = pd.DataFrame()

        for k, v in stats.items():
            self.objects_props[k] = v

        self.objects_props['orientation'] = orientation
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
        centroids = self.objects_props[['centroid-0', 'centroid-1']].to_numpy(int)
        self.objects_props['segment_labels'] = heart_slice.total_segments[tuple(centroids.T)]
        self.objects_props['fibrosis'] = heart_slice.segment_fibrosis_map[tuple(centroids.T)]
