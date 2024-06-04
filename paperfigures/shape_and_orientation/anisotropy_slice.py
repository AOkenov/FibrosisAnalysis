from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, lines

from bitis.texture.properties import DistributionEllipseBuilder

from fibrosisanalysis.parsers.stats_loader import StatsLoader
from fibrosisanalysis.slice.heart_slice import HeartSliceBuilder
from fibrosisanalysis.analysis import (
    ObjectsPropertiesBuilder,
    SegmentsPropertiesBuilder
)
from fibrosisanalysis.plots.polar_plots import PolarPlots


cmap = colors.LinearSegmentedColormap.from_list(
    'fibrosis', [(0, 'white'),
                 (0.5, '#e2a858'),
                 (1, '#990102')])

cmap_2 = colors.LinearSegmentedColormap.from_list(
    'fibrosis', [(0, '#e2a858'),
                 (1, '#990102')])


def draw_text(ax, rx, ry, text):
    ax.text(rx, ry, str(text), color='black', fontsize=7,
            bbox=dict(facecolor='white', edgecolor='white',
                      boxstyle='round, pad=0.1'),
            ha='center', va='center')


def draw_ellipse(ax, rx, ry, dist_ellipse, n_std=2):
    r = n_std * 10 * dist_ellipse.full_radius
    theta = dist_ellipse.full_theta
    y_ellipse, x_ellipse = PolarPlots.polar_to_cartesian(r, theta)
    ax.plot(rx + x_ellipse, ry + y_ellipse, color='red')


def draw_points(ax, rx, ry, objects_props, segment_index, n_std=2):
    r, theta, density = PolarPlots.sort_by_density(objects_props['axis_ratio'],
                                                   objects_props['orientation'])
    y, x = PolarPlots.polar_to_cartesian(r, theta)
    x = n_std * 10 * x + rx
    y = n_std * 10 * y + ry
    ax.scatter(x, y, c=density, s=1)


def draw_lines(fig, ax0, ax1, x0, y0, x1, y1, lw=0.5):
    fig.canvas.draw()
    transFigure = fig.transFigure.inverted()

    x0, y0 = transFigure.transform(ax0.transData.transform([x0, y0]))
    x1, y1 = transFigure.transform(ax1.transData.transform([x1, y1]))

    line = lines.Line2D((x0, x1), (y0, y1),
                        transform=fig.transFigure, lw=lw, color='black')

    fig.lines.append(line)


# path = Path(__file__).parent.parent.parent.joinpath('data')
path = Path('/Users/arstanbek/Library/CloudStorage/OneDrive-UGent/data')
path_stats = path

heart = 'E11444_LMNA'
slice_name = 'E11444_08_SC2'

# heart = 'E11444_LMNA'
# slice_name = 'E11444_10_SC2'

n_radial = 3
n_angular = 12
node_step = 10

heart_slice_builder = HeartSliceBuilder()
heart_slice_builder.build_from_file(path, heart, slice_name,
                                    n_angular, n_radial,
                                    node_step)
heart_slice = heart_slice_builder.heart_slice

# Load stats
path_slice_stats = path_stats.joinpath(heart, 'Stats', slice_name)
stats_loader = StatsLoader(path_stats)
object_stats = stats_loader.load_slice_data(path_slice_stats)

# Build objects properties
objects_props_builder = ObjectsPropertiesBuilder()
objects_props_builder.build_from_stats(object_stats)
objects_props_builder.add_slice_props(heart_slice)
objects_props = objects_props_builder.objects_props

# Build segment properties
segments_props_builder = SegmentsPropertiesBuilder()
segments_props_builder.build(heart_slice, objects_props)
segments_props = segments_props_builder.props

n_std = 2
segment_indexes = np.array([3, 15, 27])

fig = plt.figure(figsize=(8, 7))

axs = fig.subplot_mosaic([['segments', 'segments_zoom'],
                          ['sa', 'sa_zoom']])

axs['segments'].sharex(axs['sa'])
axs['segments'].sharey(axs['sa'])
axs['segments_zoom'].sharex(axs['sa_zoom'])
axs['segments_zoom'].sharey(axs['sa_zoom'])

for subplot in ['segments', 'sa']:
    ax = axs[subplot]
    index = segment_indexes[0]
    start = (index - 1) * node_step
    end = index * node_step + 1
    for spline_edge in heart_slice.spline_edges:
        ax.plot(spline_edge.full_nodes[:, 1],
                spline_edge.full_nodes[:, 0],
                'k', lw=0.5)

        ax.plot(spline_edge.nodes[start:end, 1],
                spline_edge.nodes[start:end, 0],
                'k', lw=1.)
        axs['sa_zoom'].plot(spline_edge.nodes[start:end, 1],
                            spline_edge.nodes[start:end, 0],
                            'k', lw=1.)
        axs['segments_zoom'].plot(spline_edge.nodes[start:end, 1],
                                  spline_edge.nodes[start:end, 0],
                                  'k', lw=1.)

    for k in range(n_angular):
        x = [heart_slice.spline_edges[0].ordered_nodes[k, 1],
             heart_slice.spline_edges[-1].ordered_nodes[k, 1]]
        y = [heart_slice.spline_edges[0].ordered_nodes[k, 0],
             heart_slice.spline_edges[-1].ordered_nodes[k, 0]]

        if k in [index - 1, index]:
            ax.plot(x, y, 'k', lw=1.)
            axs['sa_zoom'].plot(x, y, 'k', lw=1.)
            axs['segments_zoom'].plot(x, y, 'k', lw=1.)
        else:
            ax.plot(x, y, 'k', lw=0.5)

for _, ax in axs.items():
    ax.axis('off')
    ax.set_aspect('equal', 'box')


segment_mask = np.isin(heart_slice.total_segments, segment_indexes)

# Show image
axs['segments'].imshow(heart_slice.image, cmap=cmap, origin='lower')
im = heart_slice.image.copy()
im[~segment_mask] = 0
axs['segments_zoom'].imshow(im, cmap=cmap, origin='lower')


x_min = np.min(np.where(segment_mask)[0]) - 10
x_max = np.max(np.where(segment_mask)[0]) + 10
y_min = np.min(np.where(segment_mask)[1]) - 10
y_max = np.max(np.where(segment_mask)[1]) + 10


axs['sa_zoom'].set_xlim([y_min, y_max])
axs['sa_zoom'].set_ylim([x_min, x_max])

for segment_index in range(1, heart_slice.total_segments.max() + 1):
    segment_objects_props = objects_props[objects_props['segment_labels']
                                          == segment_index]

    r = segment_objects_props['axis_ratio'].values
    theta = segment_objects_props['orientation'].values
    r = np.concatenate([r, r])
    theta = np.concatenate([theta, theta + np.pi])
    dist_ellipse_builder = DistributionEllipseBuilder()
    dist_ellipse_builder.build(r, theta, n_std=n_std)
    dist_ellipse = dist_ellipse_builder.dist_ellipse

    coords = np.argwhere(heart_slice.total_segments == segment_index)
    ry, rx = coords.mean(axis=0)

    draw_points(axs['sa'], rx, ry, segment_objects_props, segment_index)
    draw_ellipse(axs['sa'], rx, ry, dist_ellipse)
    draw_text(axs['segments'], rx, ry, segment_index)

    if segment_index in segment_indexes:
        draw_points(axs['sa_zoom'], rx, ry, segment_objects_props,
                    segment_index)
        draw_ellipse(axs['sa_zoom'], rx, ry, dist_ellipse)
        draw_text(axs['segments_zoom'], rx, ry, segment_index)


x = segments_props['centroid-0'].values[segment_indexes - 1]
y = segments_props['centroid-1'].values[segment_indexes - 1]

edge_direction = segments_props['edge_direction'].values[segment_indexes - 1]
ellipse_width = segments_props['sa_major_axis'].values[segment_indexes - 1]
r_orientation = segments_props['relative_orientation'].values[segment_indexes - 1]

e_orientation = edge_direction - r_orientation

U = n_std * 10 * 0.5 * ellipse_width * np.cos(edge_direction)
V = n_std * 10 * 0.5 * ellipse_width * np.sin(edge_direction)
axs['sa_zoom'].quiver(y, x, V, U, scale_units='xy', scale=1, color='black')

U = n_std * 10 * 0.5 * ellipse_width * np.cos(e_orientation)
V = n_std * 10 * 0.5 * ellipse_width * np.sin(e_orientation)
axs['sa_zoom'].quiver(y, x, V, U, scale_units='xy', scale=1, color='red')

index = segment_indexes[0]

for k in [index - 1, index]:
    x = [heart_slice.spline_edges[0].ordered_nodes[k, 1],
         heart_slice.spline_edges[-1].ordered_nodes[k, 1]]
    y = [heart_slice.spline_edges[0].ordered_nodes[k, 0],
         heart_slice.spline_edges[-1].ordered_nodes[k, 0]]

    if k == index:
        draw_lines(fig, axs['segments'], axs['segments_zoom'],
                   x[1], y[1], 2840, 1290, lw=1.)
        draw_lines(fig, axs['sa'], axs['sa_zoom'], x[1], y[1], 2840, 1290,
                   lw=1.)
    else:
        draw_lines(fig, axs['segments'], axs['segments_zoom'],
                   x[1], y[1], x[1], y[1], lw=1.)
        draw_lines(fig, axs['sa'], axs['sa_zoom'], x[1], y[1], x[1], y[1],
                   lw=1.)

    draw_lines(fig, axs['segments'], axs['segments_zoom'],
               x[0], y[0], x[0], y[0], lw=1.)
    draw_lines(fig, axs['sa'], axs['sa_zoom'], x[0], y[0], x[0], y[0], lw=1.)

axs['segments'].set_title('A. Segmented Slice', loc='left', fontsize=12)
# axs['segments_zoom'].set_title('C. Segment 3, 15, 27', loc='left', fontsize=12)
axs['sa'].set_title('B. Structural Anisotropy', loc='left', fontsize=12)
# axs['sa_zoom'].set_title('D. Segment\'s SA', loc='left', fontsize=12)

plt.show()

fig.savefig('paperfigures/figures/slice_anisotropy.png', dpi=300,
            bbox_inches='tight')
