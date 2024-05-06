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

# path = Path('/Users/arstanbek/Hulk/Arstan/analysis/data')
path = Path(__file__).parent.parent.joinpath('data')
path_stats = Path(__file__).parent.parent.joinpath('data')
    # '/Users/arstanbek/Projects/fibrosis-workspace/fibrosisanalysis/examples/data')

# heart = 'E10691_RBM20'
heart = 'E11444_LMNA'
# heart = 'E10927_MYBPC3'
# heart = 'E11971_MYH7'

subdir = 'Stats'
path_ = path_stats.joinpath(heart, subdir)
files = list(path_.glob('*{}'.format('.pkl')))
files = sorted([file.stem for file in files if not file.name.startswith('.')])


n_radial_segments = 3
n_angular_segments = 12
node_step = 3

anisotropy = []
edge_directions = []
objects_orientations = []

for slice_name in tqdm(files[:]):
    # Load slice and build HeartSlice object
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
    segments_props = segments_props_builder.segments_properties

    edge_direction = segments_props.edge_direction
    objects_orientation = segments_props.objects_orientation

    # delta = (objects_orientation - edge_direction + 0.5 * np.pi) % np.pi - 0.5 * np.pi
    # objects_orientation = edge_direction + delta
    objects_orientation = segments_props.normed_objects_orientation

    edge_directions.append(edge_direction.reshape(3, -1))
    objects_orientations.append(objects_orientation.reshape(3, -1))
    anisotropy.append(segments_props.objects_anisotropy.reshape(3, -1))

anisotropy = np.concatenate(anisotropy, axis=1)
edge_direction = np.concatenate(edge_directions, axis=1)
objects_orientation = np.concatenate(objects_orientations, axis=1)


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
    segment_obj_orientation = np.degrees(objects_orientation[i] - 
                                         edge_direction[i])
    segment_obj_orientation = segment_obj_orientation[~np.isnan(segment_obj_orientation)]
    print(label)
    print('-----------------------------------------')
    print('MEAN: ', np.nanmean(segment_obj_orientation))
    print('STD: ', np.nanstd(segment_obj_orientation))
    print(np.quantile(segment_obj_orientation, [0.05, 0.25, 0.5, 0.75, 0.95]))
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

path_save = Path(__file__).parent.joinpath('figures')
# path_save.joinpath(heart).mkdir(parents=True, exist_ok=True)
# fig.savefig('paperfigures/figures/{}/fibrosis_orientation.png'.format(heart), 
#             dpi=300, bbox_inches='tight')


bins = np.linspace(1, 5, 20)
fig, axs = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(10, 4))

for i, label in enumerate(['A. Sub-Endocardium',
                           'B. Mid-Myocardium',
                           'C. Sub-Epicardium']):
    axs[i].hist(anisotropy[i], bins=bins, color=tab_colors[i])
    axs[i].set_xlabel('Structural Anisotropy')
    axs[i].set_ylabel('Number Of Segments')
    axs[i].set_xlim(1, 5)
    axs[i].set_title(label, loc='left', fontsize=14)
plt.tight_layout()
plt.show()

# fig.savefig('paperfigures/figures/{}/structural_anisotropy.png'.format(heart),
#             dpi=300, bbox_inches='tight')

plt.figure()
for i, label in enumerate(['Sub-Endocardium',
                           'Mid-Myocardium',
                           'Sub-Epicardium']):
    cumulative_frequency = np.cumsum(np.histogram(anisotropy[i], bins=bins)[0])
    plt.plot(bins[:-1], cumulative_frequency, label=label)
plt.legend()
plt.show()
