import numpy as np
import matplotlib.pyplot as plt

from fibrosisanalysis.segmentation.spline_edge import SplineEdge
from fibrosisanalysis.segmentation.angular_segments import AngularSegments
from fibrosisanalysis.segmentation.radial_segments import RadialSegments


n_angular_segments = 36
n_radial_segments = 3

r_x = {'endo': 100, 'epi': 150}
r_y = {'endo': 200, 'epi': 300}

y_0 = {'endo': int(0.8 * r_y['epi']), 'epi': r_y['epi']}

phi = np.linspace(0, 2 * np.pi, 40, endpoint=False)

shape = (2 * r_x['epi'] - 10, 2 * r_y['epi'] + 1)

spline_edges = []
for edge in ['endo', 'epi']:
    x = r_x['epi'] + r_x[edge] * np.cos(phi)
    y = y_0[edge] + r_y[edge] * np.sin(phi)

    spline_edge = SplineEdge()
    spline_edge.nodes = np.array([x, y]).T

    if edge == 'endo':
        spline_edge.nodes = np.roll(spline_edge.nodes, -10, axis=0)

    res = spline_edge.compute(10_000)
    res = spline_edge.clip_boundaries(res, shape)

    if not spline_edge.check_connectivity(res, shape):
        raise ValueError('Not enough points to draw edge')

    spline_edge.full_nodes = res
    spline_edges.append(spline_edge)


image = np.zeros(shape)

mask = RadialSegments.wall_mask(image, spline_edges)

spline_edges = AngularSegments.build(spline_edges, n_angular_segments,
                                     n_radial_segments)
spline_edges = RadialSegments.build(spline_edges, n_radial_segments)

angular_segments = AngularSegments.label(mask, spline_edges)
radial_segments = RadialSegments.label(mask, spline_edges)

fig, axs = plt.subplots(nrows=3, sharex=True, sharey=True)
axs[0].imshow(image, origin='lower')
axs[0].set_title('Mask')
axs[1].imshow(angular_segments, origin='lower')
axs[1].set_title('Radial segmentation')
axs[2].imshow(radial_segments, origin='lower')
axs[2].set_title('Angular segmentation')

for ax in axs:
    ax.set_aspect('equal')
plt.tight_layout()
plt.show()
