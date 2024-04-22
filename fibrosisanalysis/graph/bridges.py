from fibrosisprocessing.graph.elements import Elements

import networkx as nx


class Bridges(Elements):
    def __init__(self, graph, branches, leafs, cycles):
        Elements.__init__(self)

        self.make_graph(graph, leafs, branches, cycles)

    def make_graph(self, graph, leafs, branches, cycles):

        nodes = set(graph.nodes)
        for element in [leafs, branches, cycles]:
            nodes -= set(element.nodes)

        self.graph = graph.subgraph(nodes)
