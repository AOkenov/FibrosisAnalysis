from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from fibrosisanalysis.segmentation.spline_edge import SplineEdge
from fibrosisanalysis.segmentation.wall_mask import WallMask
from fibrosisanalysis.segmentation.radial_segmentation import (
    RadialSegmentation
)
from fibrosisanalysis.segmentation.angular_segmentation import (
    AngularSegmentation
)
from fibrosisanalysis.parsers import EdgeLoader, ImageLoader


path = Path('/Users/arstanbek/Hulk/Arstan/analysis/data')
edge_loader = EdgeLoader(path)
image_loader = ImageLoader(path)

image = image_loader.load_slice_data(path.joinpath('E10615_MYH7', 'Images',
                                                   'E10615_09_SC2_WABL'))
image = image.T
spline_edges = {}
for i, name in enumerate(['epi', 'endo']):
    nodes = edge_loader.load_slice_data(path.joinpath('E10615_MYH7', 'Edges',
                                                      'E10615_09_SC2_WABL_{}'.format(i)))

    spline_edge = SplineEdge()
    spline_edge.nodes = nodes
    res = spline_edge.sample_nodes(100_000)
    res = spline_edge.clip_boundaries(res, image.shape)
    spline_edge.full_nodes = res
    spline_edges[name] = spline_edge


mask = WallMask.build(image, spline_edges)
radial_labels = RadialSegmentation.build(mask, spline_edges, 6)
angular_labels = AngularSegmentation.build(mask, spline_edges, 12)

fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
axs[0, 0].imshow(image, origin='lower')
axs[0, 0].set_title('Image')
axs[0, 0].scatter(spline_edges['endo'].nodes[:, 0],
                  spline_edges['endo'].nodes[:, 1], marker='o')
axs[0, 0].scatter(spline_edges['epi'].nodes[:, 0],
                  spline_edges['epi'].nodes[:, 1], marker='o')
axs[0, 1].imshow(mask)
axs[0, 1].set_title('Mask')
axs[1, 0].imshow(radial_labels, origin='lower')
axs[1, 0].set_title('Radial segmentation')
axs[1, 1].imshow(angular_labels, origin='lower')
axs[1, 1].set_title('Angular segmentation')

# for ax in axs:
#     ax.set_aspect('equal')
plt.tight_layout()
plt.show()
