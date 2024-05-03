import pandas as pd
import numpy as np

from fibrosisanalysis.segmentation.distance import Distance


class ClustersStatsBuilder:
    def __init__(self) -> None:
        self.clusters_stats = pd.DataFrame()

    def load_from_file(self):
        pass

    def add_segment_labels(self):
        pass

    def add_distance(self, coords, edges):
        """
        Add the distance of the cluster to the endocardium."""
        d_endo, _ = Distance.shortest(edges['endo'].compute(), coords)
        d_epi, _ = Distance.shortest(edges['epi'].compute(), coords)
        self.clusters_stats['distance'] = d_endo / (d_endo + d_epi)

    def add_tangent(self, coords, spline_edge):
        """"""
        edge_coords = np.mean(spline_edge.nodes[:-1],
                              spline_edge.nodes[1:], axis=0)
        _, inds = Distance.shortest(edge_coords, coords)
        self.clusters_stats['tangent'] = spline_edge.direction[inds]

    def add_density(self, mask, intensity):
        return np.mean(intensity[mask])
