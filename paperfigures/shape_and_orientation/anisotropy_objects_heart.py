from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

path = Path(__file__).parent.parent.parent.joinpath('data')
path_stats = Path(__file__).parent.parent.parent.joinpath('data')
path_save = Path(__file__).parent.parent.joinpath('figures')

hearts = {'E10691_RBM20': 'A', 'E11444_LMNA': 'B', 'E10927_MYBPC3': 'C'}

subdir = 'Stats'

n_radial_segments = 3
n_angular_segments = 12
node_step = 3

collected_data = []

for heart, _ in hearts.items():
    path_ = path_stats.joinpath(heart, subdir)
    files = list(path_.glob('*{}'.format('.pkl')))
    files = sorted([file.stem for file in files if not file.name.startswith('.')])

    for slice_name in tqdm(files[:]):
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
        objects_props = objects_props_builder.objects_props.loc[:,
                                                                ['segment_labels',
                                                                 'relative_orientation',
                                                                 'axis_ratio']]

        objects_props['location'] = pd.cut(objects_props['segment_labels'],
                                           bins=np.linspace(0, n_angular_segments * n_radial_segments,
                                                            n_radial_segments + 1),
                                           labels=['SUB-ENDO', 'MID', 'SUB-EPI'])
        objects_props['heart'] = heart

        collected_data.append(objects_props)

df = pd.concat(collected_data, ignore_index=True)

grouped = df.groupby(['heart', 'location'], observed=True)


def degrees_formatter(x, pos):
    return f"{int(x)}°"


def percent_formatter(x, pos):
    return f"{int(x * 100)}%"


bins = np.linspace(-90, 90, 19)
colors_ = ['red', 'green', 'blue']

fig, axs = plt.subplots(ncols=4, nrows=3, sharex=True, sharey=True,
                        gridspec_kw={'width_ratios': [1, 1, 1, 0.15]},
                        figsize=(8, 8))

for i, (heart, label) in enumerate(hearts.items()):
    for j, location in enumerate(['SUB-ENDO', 'MID', 'SUB-EPI']):
        data = grouped.get_group((heart, location))
        obj_orientation = np.degrees(data['relative_orientation'].values)
        obj_orientation = obj_orientation[~np.isnan(obj_orientation)]
        count, _ = np.histogram(obj_orientation, bins=bins)
        count = count / len(obj_orientation)
        # print(heart, ': ', location)
        # print('-----------------------------------------')
        # print('MEAN: ', np.nanmean(obj_orientation))
        # print('STD: ', np.nanstd(obj_orientation))
        # print(np.quantile(obj_orientation, [0.05, 0.25, 0.5, 0.75, 0.95]))

        # axs[j, i].hist(obj_orientation, bins=bins, alpha=0.5, color=colors_[j],
        #                edgecolor='black', density=True, stacked=True)
        axs[j, i].bar(bins[:-1], count, width=10, align='edge',
                      color=colors_[j], edgecolor='black', alpha=0.5)
        axs[j, i].set_xlim(-90, 90)
        axs[j, i].set_xticks([-90, -45, 0, 45, 90])
        axs[j, i].xaxis.set_major_formatter(ticker.FuncFormatter(degrees_formatter))
        axs[j, i].yaxis.set_major_formatter(ticker.FuncFormatter(percent_formatter))

for j, location in enumerate(['SUB-ENDO', 'MID', 'SUB-EPI']):
    axs[j, 0].set_ylabel('Number of Clusters, %')
    axs[j, -1].set_axis_off()
    axs[j, -1].text(0., 0.5, location, transform=axs[j, -1].transAxes,
                    ha='center', va='center', rotation=90)

for i, (_, label) in enumerate(hearts.items()):
    axs[0, i].set_title(label, loc='left')
    axs[2, i].set_xlabel('Cluster vs. Segment\n Orientation')

    # if i == 0:
    #     continue

    # axs[0, i].sharey(axs[0, 0])
    # axs[1, i].sharey(axs[1, 0])
    # axs[2, i].sharey(axs[2, 0])

plt.subplots_adjust(top=0.9, bottom=0.1, right=0.98, left=0.1,
                    wspace=0.4, hspace=0.2)
plt.show()

fig.savefig(path_save.joinpath('clusters_orientation.png'), dpi=300,
            bbox_inches='tight')


bins = np.linspace(1, 5, 17)

fig, axs = plt.subplots(ncols=4, nrows=3, sharex=True, sharey=True,
                        gridspec_kw={'width_ratios': [1, 1, 1, 0.15]},
                        figsize=(8, 8))

for i, (heart, label) in enumerate(hearts.items()):
    for j, location in enumerate(['SUB-ENDO', 'MID', 'SUB-EPI']):
        data = grouped.get_group((heart, location))
        count, _ = np.histogram(data['axis_ratio'].values, bins=bins)
        count = count / len(data['axis_ratio'].values)

        axs[j, i].bar(bins[:-1], count, width=0.25, align='edge',
                      color=colors_[j], edgecolor='black', alpha=0.5)
        axs[j, i].set_xlim(1, 5)
        axs[j, i].yaxis.set_major_formatter(ticker.FuncFormatter(percent_formatter))

for j, location in enumerate(['SUB-ENDO', 'MID', 'SUB-EPI']):
    axs[j, 0].set_ylabel('Number of Clusters, %')
    axs[j, -1].set_axis_off()
    axs[j, -1].text(0., 0.5, location, transform=axs[j, -1].transAxes,
                    ha='center', va='center', rotation=90)

for i, (_, label) in enumerate(hearts.items()):
    axs[0, i].set_title(label, loc='left')
    axs[2, i].set_xlabel('Axis Ratio')

    # if i == 0:
    #     continue

    # axs[0, i].sharey(axs[0, 0])
    # axs[1, i].sharey(axs[1, 0])
    # axs[2, i].sharey(axs[2, 0])

plt.show()

fig.savefig(path_save.joinpath('axis_ratio.png'), dpi=300,
            bbox_inches='tight')
