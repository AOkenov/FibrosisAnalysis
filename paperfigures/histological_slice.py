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

path = Path('/Users/arstanbek/Hulk/Arstan/analysis/data')

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


x0 = [[1400, 1128], [1227, 898]]
y0 = [[2313, 2858], [2192, 2729]]

fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
ax[0].imshow(image, cmap=cmap, origin='lower')
ax[0].plot(edges['endo'].full_nodes[:, 1], edges['endo'].full_nodes[:, 0],
           '#1f77b4', lw=2)
ax[0].plot(edges['epi'].full_nodes[:, 1], edges['epi'].full_nodes[:, 0],
           '#2ca02c', lw=2)

rect = patches.Rectangle((850, 2250), 200, 200, linewidth=1, edgecolor='black',
                         facecolor='none')

ax[0].add_patch(rect)

ax[1].imshow(image, cmap=cmap)
ax[1].set_ylim(2250, 2450)
ax[1].set_xlim(850, 1050)

ax[0].set_title('A', loc='left', fontsize=16)
ax[1].set_title('B', loc='left', fontsize=16)
ax[0].axis('off')
ax[1].set_xticks([])
ax[1].set_yticks([])

plt.tight_layout()

# con = patches.ConnectionPatch(xyA=[850, 2250], 
#                               xyB=[0, 0], 
#                               coordsA="data", coordsB="data",
#                               axesA=ax[0], axesB=ax[1], color="red")
# ax[1].add_artist(con)

plt.show()

# fig.savefig('paperfigures/figures/histological_slice.png', dpi=300,
#             bbox_inches='tight')
