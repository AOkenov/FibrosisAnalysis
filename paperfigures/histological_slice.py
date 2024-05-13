from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors, patches

from fibrosisanalysis.parsers.image_loader import ImageLoader
from fibrosisanalysis.parsers.edge_loader import EdgeLoader
from fibrosisanalysis.segmentation import (
    SplineEdge
)

cmap = colors.LinearSegmentedColormap.from_list(
            'fibrosis', [(0, 'white'),
                         (0.5, '#e2a858'),
                         (1, '#990102')])

path = Path(__file__).parents[1].joinpath('data')

hearts = ['E10615_MYH7', 'E10621_ABCC9', 'E10691_RBM20', 'E10788_LMNA',
          'E10884', 'E10927_MYBPC3', 'E11442_TTN', 'E11443_LMNA',
          'E11444_LMNA', 'E11971_MYH7']

heart = 'E11444_LMNA'
slice_name = 'E11444_08_SC2'

image_loader = ImageLoader(path)
image = image_loader.load_slice_data(path.joinpath(heart, 'Images',
                                                   slice_name))

edge_loader = EdgeLoader(path)

edges = {}

for i, name in enumerate(['epi', 'endo']):
    filename = '{}_{}'.format(slice_name, i)
    nodes = edge_loader.load_slice_data(path.joinpath(heart, 'Edges',
                                                      filename))

    spline_edge = SplineEdge()
    spline_edge.nodes = nodes[:, [1, 0]]
    full_nodes = spline_edge.sample_nodes(100_000)
    spline_edge.full_nodes = spline_edge.clip_boundaries(full_nodes,
                                                         image.shape)
    edges[name] = spline_edge

n_r_segments = 6
n_a_segments = 36


# x0 = [[1400, 1128], [1227, 898]]
# y0 = [[2313, 2858], [2192, 2729]]

x0 = 3000
dx0 = 200
y0 = 1650
dy0 = 200

x1 = 3100
dx1 = 60
y1 = 1745
dy1 = 60
# x1 = [[1400, 1128], [1227, 898]]

fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(10, 4))

ax[0, 0].imshow(image, cmap=cmap, origin='lower')
ax[0, 0].plot(edges['endo'].full_nodes[:, 1], edges['endo'].full_nodes[:, 0],
           '#1f77b4', lw=2)
ax[0, 0].plot(edges['epi'].full_nodes[:, 1], edges['epi'].full_nodes[:, 0],
           '#2ca02c', lw=2)

rect = patches.Rectangle((x0, y0), dx0, dy0, linewidth=1, edgecolor='black',
                         facecolor='none')
ax[0, 0].add_patch(rect)
ax[0, 0].set_title('A', loc='left', fontsize=16)
ax[0, 0].axis('off')

ax[0, 1].imshow(image, cmap=cmap, origin='lower')
ax[0, 1].set_ylim(y0, y0 + dy0)
ax[0, 1].set_xlim(x0, x0 + dx0)

ax[0, 1].set_title('B', loc='left', fontsize=16)
ax[0, 1].set_xticks([])
ax[0, 1].set_yticks([])

rect = patches.Rectangle((x1, y1), dx1, dy1, linewidth=1, edgecolor='black',
                         facecolor='none')
ax[0, 1].add_patch(rect)

ax[0, 2].imshow(image, cmap=cmap, origin='lower')
ax[0, 2].set_ylim(y1, y1 + dy1)
ax[0, 2].set_xlim(x1, x1 + dx1)
ax[0, 2].set_title('C', loc='left', fontsize=16)
ax[0, 2].set_xticks([])
ax[0, 2].set_yticks([])

for i, label in enumerate(['D', 'E', 'F']):
    ax[1, i].imshow(image, cmap=cmap, origin='lower')
    ax[1, i].set_ylim(y1, y1 + dy1)
    ax[1, i].set_xlim(x1, x1 + dx1)
    ax[1, i].set_title(label, loc='left', fontsize=16)
    ax[1, i].set_xticks([])
    ax[1, i].set_yticks([])

plt.tight_layout()

# con = patches.ConnectionPatch(xyA=[850, 2250], 
#                               xyB=[0, 0], 
#                               coordsA="data", coordsB="data",
#                               axesA=ax[0], axesB=ax[1], color="red")
# ax[1].add_artist(con)

plt.show()

# fig.savefig('paperfigures/figures/histological_slice.png', dpi=300,
#             bbox_inches='tight')
