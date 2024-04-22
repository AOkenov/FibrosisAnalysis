from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from fibrosisanalysis.parsers.stats_loader import StatsLoader
from fibrosisanalysis.slice.heart_slice import HeartSliceBuilder
from fibrosisanalysis.analysis import (
    ObjectsPropertiesBuilder,
    SegmentsPropertiesBuilder
)


cmap = colors.LinearSegmentedColormap.from_list(
    'fibrosis', [(0, 'white'),
                 (0.5, '#e2a858'),
                 (1, '#990102')])

cmap_2 = colors.LinearSegmentedColormap.from_list(
    'fibrosis', [(0, '#e2a858'),
                 (1, '#990102')])

path = Path('/Users/arstanbek/Hulk/Arstan/analysis/data')
path_stats = Path(
    '/Users/arstanbek/Projects/fibrosis-workspace/fibrosisanalysis/examples/data')

# heart = 'E11444_LMNA'
# slice_name = 'E11444_10_SC2'

heart = 'E11971_MYH7'
slice_name = 'E11971_16_SC2_WABL'

n_radial_segments = 3
n_angular_segments = 36
node_step = 3

heart_slice_builder = HeartSliceBuilder()
heart_slice_builder.build_from_file(path, heart, slice_name,
                                    n_angular_segments, n_radial_segments,
                                    node_step)
# Load stats
stats_loader = StatsLoader(path_stats)
object_stats = stats_loader.load_slice_data(path_stats.joinpath(heart, 'Stats',
                                                                slice_name))
heart_slice_builder.add_stats(object_stats)
heart_slice = heart_slice_builder.heart_slice

objects_props_builder = ObjectsPropertiesBuilder()
objects_props_builder.build_from_stats(heart_slice)
objects_props = objects_props_builder.objects_props

n_std = 2

segments_props_builder = SegmentsPropertiesBuilder()
segments_props_builder.build(heart_slice)
segments_props = segments_props_builder.segments_properties


x = segments_props.centroids[:, 0]
y = segments_props.centroids[:, 1]

fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
axs[0].imshow(heart_slice.image, origin='lower')

markers = ['o', 's', 'v']

for i, (x_, y_) in enumerate(zip(x.reshape(3, -1), y.reshape(3, -1))):
    axs[0].scatter(y_, x_, c=np.arange(len(x_)), cmap='viridis', s=100,
                   marker=markers[i])

edge_direction = segments_props.edge_direction
objects_orientation = segments_props.objects_orientation

delta = (objects_orientation - edge_direction +
         0.5 * np.pi) % np.pi - 0.5 * np.pi
objects_orientation = edge_direction + delta

U = 300 * np.cos(edge_direction)
V = 300 * np.sin(edge_direction)
axs[0].quiver(y, x, V, U, scale_units='xy', scale=1, color='red')

U = 300 * np.cos(objects_orientation)
V = 300 * np.sin(objects_orientation)
axs[0].quiver(y, x, V, U, scale_units='xy', scale=1)


for spline_edge in heart_slice.spline_edges:
    axs[0].plot(spline_edge.full_nodes[:, 1],
                spline_edge.full_nodes[:, 0],
                'k', lw=0.5)

for k in range(len(heart_slice.spline_edges[0].ordered_nodes)):
    x = [heart_slice.spline_edges[0].ordered_nodes[k, 1],
         heart_slice.spline_edges[-1].ordered_nodes[k, 1]]
    y = [heart_slice.spline_edges[0].ordered_nodes[k, 0],
         heart_slice.spline_edges[-1].ordered_nodes[k, 0]]

    axs[0].plot(x, y, 'k', lw=0.5)

edge_direction = edge_direction.reshape(3, -1)
objects_orientation = objects_orientation.reshape(3, -1)
delta = delta.reshape(3, -1)

for i, label in enumerate(['endocardium', 'midwall', 'epicardium']):
    axs[1].scatter(edge_direction[i], objects_orientation[i],
                   c=np.arange(len(edge_direction[i])), cmap='viridis',
                   marker=markers[i], s=50, label=label)
    # plt.hist(delta[i], bins=20, alpha=0.5, label=label)

axs[1].plot([-np.pi, np.pi], [-np.pi, np.pi], 'k-')
axs[1].legend()
plt.show()
