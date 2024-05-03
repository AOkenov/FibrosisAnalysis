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

from tqdm import tqdm


cmap = colors.LinearSegmentedColormap.from_list(
    'fibrosis', [(0, 'white'),
                 (0.5, '#e2a858'),
                 (1, '#990102')])

cmap_2 = colors.LinearSegmentedColormap.from_list(
    'fibrosis', [(0, '#e2a858'),
                 (1, '#990102')])

path = Path(__file__).parent.parent.parent.joinpath('data')
path_stats = path

# heart = 'E10691_RBM20'
heart = 'E11444_LMNA'
# heart = 'E10927_MYBPC3'

subdir = 'Stats'
path_ = path_stats.joinpath(heart, subdir)
files = list(path_.glob('*{}'.format('.pkl')))
files = [file.stem for file in files if not file.name.startswith('.')]

print(path_)


n_radial_segments = 3
n_angular_segments = 1
node_step = 36

anisotropy = []
for slice_name in tqdm(files[:1]):
    heart_slice_builder = HeartSliceBuilder()
    heart_slice_builder.build_from_file(path, heart, slice_name,
                                        n_angular_segments, n_radial_segments,
                                        node_step)
    angular_segments = (heart_slice_builder.heart_slice.angular_segments > 0).astype(int)
    heart_slice_builder.heart_slice.angular_segments = angular_segments
    # Load stats
    stats_loader = StatsLoader(path_stats)
    object_stats = stats_loader.load_slice_data(
        path_stats.joinpath(heart, 'Stats', slice_name))

    heart_slice_builder.add_stats(object_stats)
    heart_slice = heart_slice_builder.heart_slice

    plt.figure()
    plt.imshow(heart_slice.angular_segments)
    plt.show()

#     objects_props_builder = ObjectsPropertiesBuilder()
#     objects_props_builder.build_from_stats(heart_slice)
#     objects_props = objects_props_builder.objects_props

#     segments_props_builder = SegmentsPropertiesBuilder()
#     segments_props_builder.build(heart_slice)
#     segments_props = segments_props_builder.segments_properties

#     anisotropy.append(segments_props.objects_anisotropy.reshape(3, -1))

# anisotropy = np.concatenate(anisotropy, axis=1)

# fig, axs = plt.subplots(ncols=3, figsize=(10, 5))
# for i, label in enumerate(['endocardium', 'midwall', 'epicardium']):
#     axs[i].scatter(edge_direction[i], objects_orientation[i], s=50, 
#                    label=label)
#     # plt.hist(delta[i], bins=20, alpha=0.5, label=label)
#     axs[i].plot([-np.pi, np.pi], [-np.pi, np.pi], 'k-')
# plt.show()


# def degrees_formatter(x, pos):
#     return f"{int(x)}°"


# bins = np.linspace(1, 5, 20)
# tab_colors = [colors.TABLEAU_COLORS[color] for color in ['tab:blue',
#                                                          'tab:orange',
#                                                          'tab:green']]
# fig, axs = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(10, 4))

# for i, label in enumerate(['A. Sub-Endocardium',
#                            'B. Mid-Myocardium',
#                            'C. Sub-Epicardium']):
#     axs[i].hist(anisotropy[i], bins=bins, color=tab_colors[i])
#     axs[i].set_xlabel('Structural Anisotropy')
#     axs[i].set_ylabel('Number Of Segments')
#     axs[i].set_xlim(1, 5)
#     # axs[i].xaxis.set_major_formatter(ticker.FuncFormatter(degrees_formatter))

#     axs[i].set_title(label, loc='left', fontsize=14)
# plt.tight_layout()
# plt.show()

# plt.figure()
# for i, label in enumerate(['Sub-Endocardium',
#                            'Mid-Myocardium',
#                            'Sub-Epicardium']):
#     cumulative_frequency = np.cumsum(np.histogram(anisotropy[i], bins=bins)[0])
#     plt.plot(bins[:-1], cumulative_frequency, label=label)
# plt.legend()
# plt.show()

# fig.savefig('paperfigures/figures/{}/structural_anisotropy.png'.format(heart),
#             dpi=300, bbox_inches='tight')
