from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

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
    ax.text(rx, ry, str(text), color='black', fontsize=8,
            bbox=dict(facecolor='white', edgecolor='white',
                      boxstyle='round, pad=0.2'),
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


path = Path(__file__).parent.parent.parent.joinpath('data')
path_stats = path

heart = 'E11444_LMNA'
slice_name = 'E11444_08_SC2'

# heart = 'E11444_LMNA'
# slice_name = 'E11444_10_SC2'

n_radial_segments = 3
n_angular_segments = 12
node_step = 3

heart_slice_builder = HeartSliceBuilder()
heart_slice_builder.build_from_file(path, heart, slice_name,
                                    n_angular_segments, n_radial_segments,
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

fig = plt.figure(figsize=(8, 7))

gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                      left=0.05, right=0.95, bottom=0.05, top=0.95,
                      wspace=0.05, hspace=0.2)
axs = []
axs.append(fig.add_subplot(gs[0, 0]))
axs.append(fig.add_subplot(gs[1, 0]))
axs.append(fig.add_subplot(gs[0, 1], sharex=axs[0], sharey=axs[0]))
axs.append(fig.add_subplot(gs[1, 1], sharex=axs[1], sharey=axs[1]))

for ax in axs[:]:
    for spline_edge in heart_slice.spline_edges:
        ax.plot(spline_edge.full_nodes[:, 1],
                spline_edge.full_nodes[:, 0],
                'k', lw=0.5)

    for k in range(len(heart_slice.spline_edges[0].ordered_nodes)):
        x = [heart_slice.spline_edges[0].ordered_nodes[k, 1],
             heart_slice.spline_edges[-1].ordered_nodes[k, 1]]
        y = [heart_slice.spline_edges[0].ordered_nodes[k, 0],
             heart_slice.spline_edges[-1].ordered_nodes[k, 0]]

        ax.plot(x, y, 'k', lw=0.5)
        ax.axis('off')
        ax.set_aspect('equal', 'box')

# Show image
axs[0].imshow(heart_slice.image, cmap=cmap, origin='lower')
axs[1].imshow(heart_slice.image, cmap=cmap, origin='lower')

segment_indexes = [3, 15, 27]

segment_mask = np.isin(heart_slice.total_segments, segment_indexes)
x_min = np.min(np.where(segment_mask)[0])
x_max = np.max(np.where(segment_mask)[0])
y_min = np.min(np.where(segment_mask)[1])
y_max = np.max(np.where(segment_mask)[1])

axs[1].set_xlim([y_min, y_max])
axs[1].set_ylim([x_min + 0.2 * (x_max - x_min), x_max])

for segment_index in range(1, heart_slice.total_segments.max() + 1):
    segment_objects_props = objects_props[objects_props['segment_labels'] == segment_index]
    dist_ellipse_builder = DistributionEllipseBuilder()
    dist_ellipse_builder.build(segment_objects_props)
    dist_ellipse = dist_ellipse_builder.dist_ellipse

    coords = np.argwhere(heart_slice.total_segments == segment_index)
    ry, rx = coords.mean(axis=0)

    draw_points(axs[2], rx, ry, segment_objects_props, segment_index)
    draw_points(axs[3], rx, ry, segment_objects_props, segment_index)

    draw_ellipse(axs[2], rx, ry, dist_ellipse)
    draw_ellipse(axs[3], rx, ry, dist_ellipse)

    draw_text(axs[0], rx, ry, segment_index)

    if segment_index in segment_indexes:
        draw_text(axs[1], rx, ry, segment_index)


x = segments_props['centroid-0']
y = segments_props['centroid-1']

edge_direction = segments_props['edge_direction']
ellipse_width = segments_props['sa_major_axis']

ellipse_orientation = (edge_direction
                       - segments_props['relative_orientation'])

U = n_std * 10 * 0.5 * ellipse_width * np.cos(edge_direction)
V = n_std * 10 * 0.5 * ellipse_width * np.sin(edge_direction)
axs[3].quiver(y, x, V, U, scale_units='xy', scale=1, color='black')

U = n_std * 10 * 0.5 * ellipse_width * np.cos(ellipse_orientation)
V = n_std * 10 * 0.5 * ellipse_width * np.sin(ellipse_orientation)
axs[3].quiver(y, x, V, U, scale_units='xy', scale=1, color='red')


axs[0].set_title('A', loc='left', fontsize=14)
axs[1].set_title('C', loc='left', fontsize=14)
axs[2].set_title('B', loc='left', fontsize=14)
axs[3].set_title('D', loc='left',
                 fontsize=14)

plt.show()

# fig.savefig('paperfigures/figures/slice_anisotropy.png', dpi=300,
#             bbox_inches='tight')
