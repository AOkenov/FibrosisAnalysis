import numpy as np
import matplotlib.pyplot as plt

from fibrosisanalysis.segmentation.spline_edge import SplineEdge
from fibrosisanalysis.segmentation.distance import Distance

r_x = {'endo': 200, 'epi': 300}
r_y = {'endo': 100, 'epi': 150}

phi = np.linspace(0, 2 * np.pi, 40, endpoint=False)

shape = (2 * r_x['epi'] - 10, 2 * r_y['epi'] + 1)

n_nodes = {'endo': 50, 'epi': 10}

spline_edges = {}
for edge in ['endo', 'epi']:
    x = r_x['epi'] + r_x[edge] * np.cos(phi)
    y = r_y['epi'] + r_y[edge] * np.sin(phi)

    spline_edge = SplineEdge()
    nodes = np.array([x, y]).T

    spline_edge.nodes = spline_edge.clip_boundaries(nodes, shape)

    res = spline_edge.sample_nodes(n_nodes[edge])
    res = spline_edge.clip_boundaries(res, shape)

    spline_edge.full_nodes = res
    spline_edges[edge] = spline_edge


_, inds = Distance.shortest(spline_edges['endo'].full_nodes,
                            spline_edges['epi'].full_nodes)

x_endo, y_endo = spline_edges['endo'].full_nodes[inds].T
x_epi, y_epi = spline_edges['epi'].full_nodes.T


plt.figure()

for edge, spline_edge in spline_edges.items():
    x, y = spline_edge.full_nodes.T
    plt.scatter(x, y, c=np.arange(len(x)), cmap='viridis', marker='o')

x, y = x_endo, y_endo
plt.scatter(x, y, c=np.arange(len(x)), cmap='plasma', marker='x')

x, y = x_epi, y_epi
plt.scatter(x, y, c=np.arange(len(x)), cmap='plasma', marker='x')

plt.show()
