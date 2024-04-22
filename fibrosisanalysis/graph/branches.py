from fibrosisprocessing.graph.elements import Elements

import networkx as nx
import numpy as np
import itertools
from scipy import spatial


class Branches(Elements):
    def __init__(self, graph):
        Elements.__init__(self)
        self.make_graph(graph)

    def make_graph(self, graph):
        degree = dict(graph.degree())
        nodes = [node for node, deg in degree.items() if deg > 2]

        self.graph = graph.subgraph(nodes)
