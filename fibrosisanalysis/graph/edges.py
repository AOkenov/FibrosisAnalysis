from fibrosisprocessing.graph.elements import Elements


class Edges(Elements):
    def __init__(self, graph, branches):
        Elements.__init__(self)

        self.make_graph(graph, branches)

    def make_graph(self, graph, branches):

        nodes = set(graph.nodes)
        nodes -= set(branches.nodes)

        self.graph = graph.subgraph(nodes)
