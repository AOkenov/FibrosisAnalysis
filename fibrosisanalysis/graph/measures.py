import networkx as nx


class Measures:
    def __init__(self, graph):
        self.graph = graph

    def multi_shortest_path(self, sources, target):
        return nx.multi_source_dijkstra(self.graph, sources=sources, target=target)

    def single_shortest_path(self, source, target):
        return nx.dijkstra_path_length(self.graph, source=source, target=target)
