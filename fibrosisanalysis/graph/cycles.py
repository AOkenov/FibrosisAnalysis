from fibrosisprocessing.graph.elements import Elements

import networkx as nx


class Cycles(Elements):
    def __init__(self, graph):
        Elements.__init__(self)

        self.make_graph(graph)

    def make_graph(self, graph):
        self.cycles = [cycle for cycle in list(
            nx.minimum_cycle_basis(graph)) if len(cycle) > 3]

    @property
    def number(self):
        return len(self.nodes)

    @property
    def connected_nodes(self):
        return self.cycles

    @property
    def length(self):
        return [len(cycle) for cycle in self.cycles]

    def diameter(self):
        return
