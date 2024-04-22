import networkx as nx
import itertools


class Filters:
    def __init__(self):
        pass

    @staticmethod
    def reduce(graph):
        degree = dict(graph.degree())
        nodes = [node for node, deg in degree.items() if deg == 2]
        edge_graph = graph.subgraph(nodes)

        degree = dict(graph.degree())
        nodes = [node for node, deg in degree.items() if deg != 2]
        reduced_graph = nx.MultiGraph(graph.subgraph(nodes).copy())

        route = {}
        for edge in reduced_graph.edges:
            route[edge] = list(edge[:-1])

        nx.set_edge_attributes(reduced_graph, route, name='route')

        for line in list(nx.connected_components(edge_graph)):
            nodes = []
            for ll in line:
                for i, _ in graph[ll].items():
                    if graph.degree(i) != 2:
                        nodes.append(i)

            route = nodes + list(line)
            g = graph.subgraph(route)
            reduced_graph.add_edge(*nodes, length=g.size(weight="length"),
                                   route=route)

        return reduced_graph

    @staticmethod
    def remove_diagonal_edges(graph):
        '''Remove edges which form triangles on branching points
        '''
        degree = dict(graph.degree())
        nodes = [node for node, deg in degree.items() if deg > 2]

        subgraph = graph.subgraph(nodes).copy()

        cycles = list(nx.cycle_basis(subgraph))

        for nodes in cycles:
            for u, v in itertools.combinations(nodes, 2):
                weight = graph.get_edge_data(u, v,
                                             default={'length': 0}
                                             )['length']
                if weight != 1 and graph.has_edge(u, v):
                    graph.remove_edge(u, v)
        return graph

    @staticmethod
    def add_edge_attributes(graph, func, name, *args):
        attrs = {}
        for u, v, k, data in graph.edges(keys=True, data=True):
            attrs[(u, v, k)] = func(data, *args)

        nx.set_edge_attributes(graph, attrs, name=name)

        return graph

    @staticmethod
    def add_node_attributes(graph, attrs, name):
        nodes = graph.nodes
        attrs = [tuple((x, y)) for x, y in attrs.tolist()]

        nx.set_node_attributes(graph, dict(zip(nodes, attrs)), name=name)
        return graph
