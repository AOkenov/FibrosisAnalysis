from fibrosisprocessing.graph.branches import Branches
from fibrosisprocessing.graph.edges import Edges
from fibrosisprocessing.graph.leafs import Leafs

import networkx as nx
import numpy as np

from skimage import graph as skgraph
from skimage import morphology


class Image:
    def __init__(self):
        pass

    @staticmethod
    def properties(graph):
        branches = Branches(graph)
        edges = Edges(graph, branches)
        leafs = Leafs(graph, branches)

        self.props = {'graph length': graph.number_of_nodes,
                      'branches number': branches.number,
                      'edges number': edges.number,
                      'edges length': edges.length,
                      'leafs number': leafs.number,
                      'leafs length': leafs.length}

        props = {}
        for k, prop in self.props.items():
            props[k] = prop()

        return props

    @staticmethod
    def mask(image):
        return morphology.remove_small_holes(image > 0, area_threshold=3)

    @staticmethod
    def medial_axis(image, skelet):
        _, distance = morphology.medial_axis(image, return_distance=True)
        return skelet * distance

    @staticmethod
    def skelet(image):
        return morphology.skeletonize(image, method='lee') / 255

    @staticmethod
    def coords(skelet):
        return np.argwhere(skelet > 0)

    @staticmethod
    def graph(skelet):
        matrix, _ = skgraph.pixel_graph(skelet, mask=skelet > 0,
                                        connectivity=2)
        graph = nx.from_scipy_sparse_array(matrix, edge_attribute='length')
        return graph
