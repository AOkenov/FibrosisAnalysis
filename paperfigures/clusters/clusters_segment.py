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
from fibrosisanalysis.analysis.objects_properties import ObjectsPropertiesBuilder


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

y1 = 2440
dy1 = 60
x1 = 990
dx1 = 60

y0 = y1
dy0 = dy1
x0 = x1
dx0 = dx1

edge_colors = ['#1f77b4', '#2ca02c']


def clear_image(image, min_size=10):
    mask = image == 2
    mask = morphology.remove_small_objects(mask, min_size)
    mask = segmentation.clear_border(mask)
    image[(mask == 0) & (image > 0)] = 1
    return image


def draw_convex_hull(ax, labeled, colors):
    for i in range(1, labeled.max() + 1):
        color = colors[i - 1]
        coords = np.argwhere(labeled == i)
        hull = spatial.ConvexHull(coords)
        hull_vertices = list(hull.vertices) + [hull.vertices[0]]
        hull_coords = coords[hull_vertices]
        ax.plot(hull_coords[:, 1], hull_coords[:, 0], color=color, lw=1)


def draw_ellipses(ax, props, colors):
    for i, label in enumerate(props['label']):
        color = colors[label - 1]
        width = props['major_axis_length'][i]
        height = props['minor_axis_length'][i]
        alpha = props['orientation'][i]
        centroids = props[['centroid-0', 'centroid-1']].to_numpy(int)
        xy = centroids[i]

        res = PolarPlots.rotated_ellipse(width, height, 0.5 * np.pi - alpha)
        y, x = PolarPlots.polar_to_cartesian(*res)
        y += xy[1]
        x += xy[0]
        ax.plot(y, x, color=color, lw=1)


def draw_lines(fig, ax0, ax1, x0, y0, x1, y1):
    fig.canvas.draw()
    transFigure = fig.transFigure.inverted()

    x0, y0 = transFigure.transform(ax0.transData.transform([x0, y0]))
    x1, y1 = transFigure.transform(ax1.transData.transform([x1, y1]))

    line = mpl.lines.Line2D((x0, x1), (y0, y1),
                            transform=fig.transFigure, lw=1, color='black')

    fig.lines.append(line)


# fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(8, 8))
# fig = plt.figure(layout="constrained", figsize=(8, 8))
fig = plt.figure(figsize=(8, 2.5))
axs = fig.subplot_mosaic([
    ["zoom0", "label",
     "convex", "ellipse"],
])

image_0 = image[y0: y0 + dy0, x0: x0 + dx0]
axs["zoom0"].imshow(image_0, cmap=cmap, origin='lower', aspect='equal',
                    vmin=0, vmax=2)

axs["zoom0"].set_title('A', loc='left', fontsize=16)
axs["zoom0"].set_xticks([])
axs["zoom0"].set_yticks([])

image_1 = image[y1: y1 + dy1, x1: x1 + dx1]
image_1 = clear_image(image_1, 10)

colors = plt.cm.tab20(np.linspace(0, 1, 20))


labeled = morphology.label(image_1 == 2, connectivity=1)
labeled_masked = np.ma.masked_where(labeled == 0, labeled)
axs["label"].imshow(labeled_masked, cmap='tab20', origin='lower',
                    aspect='equal', vmin=1, vmax=21)
axs["label"].set_title('B', loc='left', fontsize=16)
axs["label"].set_xticks([])
axs["label"].set_yticks([])

axs["convex"].imshow(labeled_masked, cmap='tab20', origin='lower',
                     aspect='equal', vmin=1, vmax=21)
draw_convex_hull(axs["convex"], labeled, ["black"] * 20)
axs["convex"].set_title('C', loc='left', fontsize=16)
axs["convex"].set_xticks([])
axs["convex"].set_yticks([])

objects_props_builder = ObjectsPropertiesBuilder()
objects_props = objects_props_builder.build_from_segment(labeled > 0)

axs["ellipse"].imshow(labeled_masked, cmap='tab20', origin='lower',
                      aspect='equal', vmin=1, vmax=21)
draw_ellipses(axs["ellipse"], objects_props, ['black'] * 20)
axs["ellipse"].set_title('D', loc='left', fontsize=16)
axs["ellipse"].set_xticks([])
axs["ellipse"].set_yticks([])

plt.tight_layout()
plt.show()

fig.savefig('paperfigures/figures/fibrotic_clusters.png', dpi=300,
            bbox_inches='tight')
