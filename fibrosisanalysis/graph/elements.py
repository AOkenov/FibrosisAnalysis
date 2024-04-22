import networkx as nx
import numpy as np
from itertools import chain


class Elements:
    def __init__(self):
        self.graph = None

    @property
    def connected_nodes(self):
        return list(nx.connected_components(self.graph))

    @property
    def nodes(self):
        return list(chain.from_iterable(self.connected_nodes))

    def number(self):
        return nx.number_connected_components(self.graph)

    def length(self):
        connected_nodes = self.connected_nodes
        if len(connected_nodes) == 0:
            return 0
        return np.mean([len(nodes) for nodes in connected_nodes])
