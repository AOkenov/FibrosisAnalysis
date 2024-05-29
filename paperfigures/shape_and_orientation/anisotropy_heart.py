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
path = Path(__file__).parent.parent.parent.joinpath('data')
path_stats = Path(__file__).parent.parent.parent.joinpath('data')
path_save = Path(__file__).parent.parent.joinpath('figures')

hearts = {'E10691_RBM20': 'A', 'E11444_LMNA': 'B', 'E10927_MYBPC3': 'C'}

subdir = 'Stats'

n_radial_segments = 3
n_angular_segments = 12
node_step = 3

anisotropy = {}
edge_directions = {}
objects_orientations = {}

for heart, _ in hearts.items():
    path_ = path_stats.joinpath(heart, subdir)
    files = list(path_.glob('*{}'.format('.pkl')))
    files = sorted([file.stem for file in files if not file.name.startswith('.')])

    anisotropy_ = []
    edge_directions_ = []
    objects_orientations_ = []
    for slice_name in tqdm(files[:2]):
        # Load slice and build HeartSlice object
        heart_slice_builder = HeartSliceBuilder()
        heart_slice_builder.build_from_file(path, heart, slice_name,
                                            n_angular_segments,
                                            n_radial_segments, node_step)
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

        edge_direction = segments_props['edge_direction'].values
        objects_orientation = segments_props['relative_orientation'].values
        sa = segments_props['structural_anisotropy'].values

        edge_directions_.append(edge_direction.reshape(3, -1))
        objects_orientations_.append(objects_orientation.reshape(3, -1))
        anisotropy_.append(sa.reshape(3, -1))

    anisotropy[heart] = np.concatenate(anisotropy_, axis=1)
    edge_directions[heart] = np.concatenate(edge_directions_, axis=1)
    objects_orientations[heart] = np.concatenate(objects_orientations_, axis=1)


def degrees_formatter(x, pos):
    return f"{int(x)}°"


bins = np.linspace(-90, 90, 20)
tab_colors = [colors.TABLEAU_COLORS[color] for color in ['tab:blue',
                                                         'tab:orange',
                                                         'tab:green']]
colors_ = ['red', 'green', 'blue']

fig, axs = plt.subplots(ncols=4, nrows=3, sharex=True, sharey=True,
                        gridspec_kw={'width_ratios': [1, 1, 1, 0.15]},
                        figsize=(8, 8))

for i, (heart, label) in enumerate(hearts.items()):
    for j, location in enumerate(['SUB-ENDO', 'MID', 'SUB-EPI']):
        obj_orientation = np.degrees(objects_orientations[heart][j])
        obj_orientation = obj_orientation[~np.isnan(obj_orientation)]
        print(heart, ': ', location)
        print('-----------------------------------------')
        print('MEAN: ', np.nanmean(obj_orientation))
        print('STD: ', np.nanstd(obj_orientation))
        print(np.quantile(obj_orientation, [0.05, 0.25, 0.5, 0.75, 0.95]))

        axs[j, i].hist(obj_orientation, bins=bins, alpha=0.5, color=colors_[j],
                       edgecolor='black')
        axs[j, i].set_xlim(-90, 90)
        axs[j, i].set_xticks([-90, -45, 0, 45, 90])
        axs[j, i].xaxis.set_major_formatter(ticker.FuncFormatter(degrees_formatter))

for j, location in enumerate(['SUB-ENDO', 'MID', 'SUB-EPI']):
    axs[j, 0].set_ylabel('Number of Segments')
    axs[j, -1].set_axis_off()
    axs[j, -1].text(0., 0.5, location, transform=axs[j, -1].transAxes,
                    ha='center', va='center', rotation=90)

for i, (_, label) in enumerate(hearts.items()):
    axs[0, i].set_title(label, loc='left')
    axs[2, i].set_xlabel('Fibrosis vs. Segment\n Orientation')

    # if i == 0:
    #     continue

    # axs[0, i].sharey(axs[0, 0])
    # axs[1, i].sharey(axs[1, 0])
    # axs[2, i].sharey(axs[2, 0])

plt.subplots_adjust(top=0.9, bottom=0.1, right=0.98, left=0.1,
                    wspace=0.4, hspace=0.2)
plt.show()

# fig.savefig(path_save.joinpath('orientation.png'), dpi=300,
#             bbox_inches='tight')


bins = np.linspace(1, 5, 20)

fig, axs = plt.subplots(ncols=4, nrows=3, sharex=True, sharey=True,
                        gridspec_kw={'width_ratios': [1, 1, 1, 0.15]},
                        figsize=(8, 8))

for i, (heart, label) in enumerate(hearts.items()):
    for j, location in enumerate(['SUB-ENDO', 'MID', 'SUB-EPI']):
        axs[j, i].hist(anisotropy[heart][j], bins=bins, color=colors_[j],
                       alpha=0.5, edgecolor='black')
        axs[j, i].set_xlim(1, 5)

for j, location in enumerate(['SUB-ENDO', 'MID', 'SUB-EPI']):
    axs[j, 0].set_ylabel('Number of Segments')
    axs[j, -1].set_axis_off()
    axs[j, -1].text(0., 0.5, location, transform=axs[j, -1].transAxes,
                    ha='center', va='center', rotation=90)

for i, (_, label) in enumerate(hearts.items()):
    axs[0, i].set_title(label, loc='left')
    axs[2, i].set_xlabel('Structural Anisotropy')

    # if i == 0:
    #     continue

    # axs[0, i].sharey(axs[0, 0])
    # axs[1, i].sharey(axs[1, 0])
    # axs[2, i].sharey(axs[2, 0])

plt.show()

# fig.savefig(path_save.joinpath('structural_anisotropy.png'), dpi=300,
#             bbox_inches='tight')
