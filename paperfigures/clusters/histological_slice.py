from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, patches
import matplotlib as mpl
from skimage import morphology, segmentation
from scipy import spatial

from fibrosisanalysis.parsers.image_loader import ImageLoader
from fibrosisanalysis.parsers.edge_loader import EdgeLoader
from fibrosisanalysis.segmentation import (
    SplineEdge
)
from fibrosisanalysis.plots.polar_plots import PolarPlots


cmap = colors.LinearSegmentedColormap.from_list(
    'fibrosis', [(0, 'white'),
                 (0.5, '#e2a858'),
                 (1, '#990102')])

# path = Path(__file__).parents[1].joinpath('data')
path = Path('/Users/arstanbek/Library/CloudStorage/OneDrive-UGent/data')

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

# plt.figure()
# plt.imshow(image[2400:2500, 970:1070], origin='lower', cmap=cmap)
# plt.show()

y0 = 2300
dy0 = 300
x0 = 870
dx0 = 300

y1 = 2400
dy1 = 100
x1 = 970
dx1 = 100

y0 = y1
dy0 = dy1
x0 = x1
dx0 = dx1

edge_colors = ['#1f77b4', '#2ca02c']


def draw_lines(fig, ax0, ax1, x0, y0, x1, y1):
    fig.canvas.draw()
    transFigure = fig.transFigure.inverted()

    x0, y0 = transFigure.transform(ax0.transData.transform([x0, y0]))
    x1, y1 = transFigure.transform(ax1.transData.transform([x1, y1]))

    line = mpl.lines.Line2D((x0, x1), (y0, y1),
                            transform=fig.transFigure, lw=0.5, color='black')

    fig.lines.append(line)


# fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(8, 8))
# fig = plt.figure(layout="constrained", figsize=(8, 8))
fig = plt.figure(figsize=(8, 4))
axs = fig.subplot_mosaic([
    ["slice", "slice", "slice", "zoom0", "zoom0"],
    ["slice", "slice", "slice", "zoom0", "zoom0"],
    ["slice", "slice", "slice", "empty", "empty"],
])

axs["empty"].axis('off')

axs["slice"].imshow(image, cmap=cmap, origin='lower', aspect='equal')
axs["slice"].plot(edges['endo'].full_nodes[:, 1],
                  edges['endo'].full_nodes[:, 0],
                  color=edge_colors[0], lw=2)
axs["slice"].plot(edges['epi'].full_nodes[:, 1], edges['epi'].full_nodes[:, 0],
                  edge_colors[1], lw=2)

rect = patches.Rectangle((x0, y0), dx0, dy0, linewidth=1, edgecolor='black',
                         facecolor='none')
axs["slice"].add_patch(rect)
# axs["slice"].set_title('Histological Slice', loc='center', fontsize=12)
axs["slice"].axis('off')

image_0 = image[y0: y0 + dy0, x0: x0 + dx0]
axs["zoom0"].imshow(image_0, cmap=cmap, origin='lower', aspect='equal',
                    vmin=0, vmax=2)
# axs["zoom0"].set_ylim(y0, y0 + dy0)
# axs["zoom0"].set_xlim(x0, x0 + dx0)

# axs["zoom0"].set_title('B', loc='left', fontsize=16)
axs["zoom0"].set_xticks([])
axs["zoom0"].set_yticks([])

rect = patches.Rectangle((x1-x0, y1-y0), dx1, dy1, linewidth=1,
                         edgecolor='black', facecolor='none')
# axs["zoom0"].add_patch(rect)
draw_lines(fig, axs["slice"], axs["zoom0"], x0, y0, 0, 0)
draw_lines(fig, axs["slice"], axs["zoom0"], x0, y0 + dy0, 0, dy0)

plt.show()

fig.savefig('paperfigures/figures/histological_slice.png', dpi=300,
            bbox_inches='tight')
