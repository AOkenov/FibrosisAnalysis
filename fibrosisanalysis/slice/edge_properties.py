import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure

from fibrosisprocessing.segmentation.axial_lines import AxialLines
from fibrosisprocessing.tools import Distance


class EdgeProperties:
    def __init__(self):
        pass

    @staticmethod
    def extract_edge(mask, width=5, label=1):
        mask = mask.copy()
        structure = ndimage.generate_binary_structure(2, connectivity=1)
        eroded = ndimage.binary_erosion(mask, structure=structure,
                                        iterations=width)
        mask[eroded > 0] = 0

        labels, _ = ndimage.label(mask, structure=np.ones((3, 3)))
        return labels == label, eroded

    @staticmethod
    def separate_segments(edge, mask, number_of_segments, width=3):

        edge_map, eroded = EdgeProperties.extract_edge(mask.copy(), width=width)
        edge_eroded, _ = EdgeProperties.extract_edge(eroded, width=1)

        nodes1 = edge.compute(number_of_segments)
        nodes2 = np.argwhere(edge_eroded > 0)
        nodes1, nodes2 = EdgeProperties.find_nodes(nodes1, nodes2)

        return AxialLines.label(edge_map, nodes1, nodes2)

    @staticmethod
    def find_nodes(nodes1, nodes2):
        d, inds = Distance.shortest(nodes2, nodes1)
        return nodes1, nodes2[inds]

    @staticmethod
    def compute(mask, edge, props_list, number_of_segments=360,
                segments_width=3):
        segments = EdgeProperties.separate_segments(edge, mask,
                                                    number_of_segments,
                                                    width=segments_width)
        props = measure.regionprops_table(segments, properties=props_list)
        props = pd.DataFrame(props)
        orientation = props['orientation'].to_numpy()
        props['orientation'] = EdgeProperties._fix_errors(props['label'],
                                                          orientation)
        return props

    @staticmethod
    def _fix_errors(label, orientation):
        '''Fix orientation error for pi / 4'''
        orientation = orientation.copy()
        orientation_diff = np.abs(np.diff(orientation, append=orientation[0]))

        error_inds = np.where((orientation_diff > 0.25 * np.pi) &
                              (orientation_diff < 0.75 * np.pi))[0]

        if error_inds.size == 0:
            return orientation

        error_inds = np.split(error_inds, error_inds.size / 2)

        for inds in error_inds:
            label_mask = label.between(inds[0] + 2, inds[1] + 2,
                                       inclusive='left')
            orientation[label_mask] = np.where(orientation[label_mask] < 0,
                                               0.5 * np.pi +
                                               orientation[label_mask],
                                               - 0.5 * np.pi +
                                               orientation[label_mask])
        return orientation
