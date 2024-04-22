from fibrosisprocessing.graph.elements import Elements

import networkx as nx


class Leafs(Elements):
    def __init__(self, graph, branches):
        Elements.__init__(self)

        self.make_graph(graph, branches)

    def make_graph(self, graph, branches):
        if len(branches.nodes) == 0:
            self.graph = nx.Graph()
            return

        degree = dict(graph.degree())
        leafs = [node for node, deg in degree.items() if deg == 1]

        nodes = []
        for leaf in leafs:
            _, lnodes = nx.multi_source_dijkstra(graph, sources=branches.nodes,
                                                 target=leaf)
            nodes += lnodes

        nodes = set(nodes)
        nodes -= set(branches.nodes)

        self.graph = graph.subgraph(nodes)
