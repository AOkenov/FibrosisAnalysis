from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, ticker

from fibrosisanalysis.parsers.stats_loader import StatsLoader
from fibrosisanalysis.slice.heart_slice import HeartSliceBuilder
from fibrosisanalysis.analysis import (
    ObjectsPropertiesBuilder,
    SegmentsPropertiesBuilder
)

from tqdm import tqdm


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
heart = 'E10691_RBM20'


subdir = 'Stats'
path_ = path_stats.joinpath(heart, subdir)
files = list(path_.glob('*{}'.format('.pkl')))
files = [file.stem for file in files if not file.name.startswith('.')]

n_radial_segments = 3
n_angular_segments = 36
node_step = 3

edge_directions = []
objects_orientations = []

for slice_name in tqdm(files[:]):
    heart_slice_builder = HeartSliceBuilder()
    heart_slice_builder.build_from_file(path, heart, slice_name,
                                        n_angular_segments, n_radial_segments,
                                        node_step)
    # Load stats
    stats_loader = StatsLoader(path_stats)
    object_stats = stats_loader.load_slice_data(
        path_stats.joinpath(heart, 'Stats', slice_name))

    heart_slice_builder.add_stats(object_stats)
    heart_slice = heart_slice_builder.heart_slice

    objects_props_builder = ObjectsPropertiesBuilder()
    objects_props_builder.build_from_stats(heart_slice)
    objects_props = objects_props_builder.objects_props

    segments_props_builder = SegmentsPropertiesBuilder()
    segments_props_builder.build(heart_slice)
    segments_props = segments_props_builder.segments_properties

    edge_direction = segments_props.edge_direction
    objects_orientation = segments_props.objects_orientation

    delta = (objects_orientation - edge_direction + 0.5 * np.pi) % np.pi - 0.5 * np.pi
    objects_orientation = edge_direction + delta

    edge_direction = edge_direction.reshape(3, -1)
    objects_orientation = objects_orientation.reshape(3, -1)

    edge_directions.append(edge_direction)
    objects_orientations.append(objects_orientation)

edge_direction = np.concatenate(edge_directions, axis=1)
objects_orientation = np.concatenate(objects_orientations, axis=1)

# fig, axs = plt.subplots(ncols=3, figsize=(10, 5))
# for i, label in enumerate(['endocardium', 'midwall', 'epicardium']):
#     axs[i].scatter(edge_direction[i], objects_orientation[i], s=50, 
#                    label=label)
#     # plt.hist(delta[i], bins=20, alpha=0.5, label=label)
#     axs[i].plot([-np.pi, np.pi], [-np.pi, np.pi], 'k-')
# plt.show()


def degrees_formatter(x, pos):
    return f"{int(x)}°"


bins = np.linspace(-90, 90, 20)
tab_colors = [colors.TABLEAU_COLORS[color] for color in ['tab:blue',
                                                         'tab:orange',
                                                         'tab:green']]
fig, axs = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(10, 4))

for i, label in enumerate(['A. Sub-Endocardium',
                           'B. Mid-Myocardium',
                           'C. Sub-Epicardium']):
    axs[i].hist(np.degrees(objects_orientation[i] - edge_direction[i]),
                bins=bins, alpha=1,
                color=tab_colors[i])
    axs[i].set_xlabel('Fibrosis vs. Segment\n Orientation')
    axs[i].set_ylabel('Number Of Segments')
    axs[i].set_xlim(-90, 90)
    axs[i].set_xticks([-90, -45, 0, 45, 90])
    axs[i].xaxis.set_major_formatter(ticker.FuncFormatter(degrees_formatter))

    axs[i].set_title(label, loc='left', fontsize=14)
plt.tight_layout()
plt.show()

fig.savefig('paperfigures/figures/fibrosis_orientation.png', dpi=300,
            bbox_inches='tight')
