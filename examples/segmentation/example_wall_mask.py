import numpy as np
import matplotlib.pyplot as plt

from fibrosisanalysis.segmentation.spline_edge import SplineEdge
from fibrosisanalysis.segmentation import RadialSegments


r_x = {'endo': 200, 'epi': 300}
r_y = {'endo': 100, 'epi': 150}

phi = np.linspace(0, 2 * np.pi, 40, endpoint=False)

shape = (2 * r_x['epi'] - 10, 2 * r_y['epi'] + 1)

spline_edges = []
for edge in ['endo', 'epi']:
    x = r_x['epi'] + r_x[edge] * np.cos(phi)
    y = r_y['epi'] + r_y[edge] * np.sin(phi)

    spline_edge = SplineEdge()
    nodes = np.array([x, y]).T

    spline_edge.nodes = nodes
    spline_edge.ordered_nodes = spline_edge.clip_boundaries(nodes, shape)

    res = spline_edge.compute(10_000)
    res = spline_edge.clip_boundaries(res, shape)

    if not spline_edge.check_connectivity(res, shape):
        raise ValueError('Not enough points to draw edge')

    spline_edge.full_nodes = res
    spline_edges.append(spline_edge)

image = np.zeros(shape)

image = RadialSegments.wall_mask(image, spline_edges)

# image[tuple(res.T)] = 1

nodes = spline_edge.ordered_nodes

plt.figure()
plt.imshow(image.T, origin='lower')
plt.scatter(x, y, c=np.arange(len(x)), cmap='viridis', marker='o')
plt.scatter(nodes[:, 0], nodes[:, 1], c=np.arange(len(x)), cmap='viridis',
            marker='x')
plt.show()
