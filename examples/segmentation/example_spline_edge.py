import numpy as np
import matplotlib.pyplot as plt

from fibrosisanalysis.segmentation.spline_edge import SplineEdge
from fibrosisanalysis.segmentation.distance import Distance

r_x = 300
r_y = 200

x0 = r_x
y0 = r_y
phi = np.linspace(0, 2 * np.pi, 40, endpoint=False)

# phi = np.concatenate((np.linspace(0, np.pi, 30),
#                       np.linspace(np.pi, 2 * np.pi, 11, endpoint=False)[1:]))

shape = (2 * r_x + 1, 2 * r_y + 1)

x = x0 + r_x * np.cos(phi)
y = y0 + r_y * np.sin(phi)

spline_edge = SplineEdge()
nodes = np.array([x, y]).T
spline_edge.nodes = nodes

res = spline_edge.sample_nodes(10)

res = spline_edge.clip_boundaries(res, shape)

print('{} nodes are sampled'.format(len(res)))

U, V = spline_edge.direction(res).T

# if not spline_edge.check_connectivity(res, shape):
#     raise ValueError('Not enough points to draw edge')

fig, axs = plt.subplots(ncols=2, nrows=2)
axs[0, 1].hist(Distance.between(res, np.roll(res, -1, axis=0)))
axs[0, 1].set_xlim([130, 190])
axs[0, 1].set_xlabel('Distance')

axs[1, 1].hist(Distance.between(nodes[::4], np.roll(nodes[::4], -1, axis=0)))
axs[1, 1].set_xlim([130, 190])
axs[1, 1].set_xlabel('Distance')

axs[0, 0].scatter(res[:, 0], res[:, 1], c=np.arange(len(res)),
                  label='sampled (N={})'.format(len(res)))
axs[0, 0].scatter(x, y, c=np.arange(len(x)), marker='x',
                  label='original (N={})'.format(len(x)))
axs[0, 0].quiver(res[:, 0], res[:, 1], U, V, angles='xy', scale_units='xy',
                 scale=1, label='direction')

axs[1, 0].scatter(x[::4], y[::4], c=np.arange(len(x[::4])), marker='o',
                  label='original (N={})'.format(len(x[::4])))
axs[1, 0].scatter(x, y, c=np.arange(len(x)), marker='x',
                  label='original (N={})'.format(len(x)))
axs[0, 0].legend()
axs[1, 0].legend()
plt.tight_layout()
plt.show()
