from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import colors, lines

from bitis.texture.properties import DistributionEllipseBuilder
from bitis.texture.properties import PatternPropertiesBuilder

from fibrosisanalysis.segmentation import SquaredSegments
from fibrosisanalysis.parsers import StatsLoader
from fibrosisanalysis.slice import HeartSliceBuilder
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


# path = Path(__file__).parent.parent.parent.joinpath('data')
path = Path('/Users/arstanbek/Library/CloudStorage/OneDrive-UGent/data')
path_stats = path

heart = 'E11444_LMNA'
slice_name = 'E11444_08_SC2'

# heart = 'E11443_LMNA'
# slice_name = 'E11443_06_SC2_NABL'

n_radial = 1
n_angular = 12
node_step = 10

heart_slice_builder = HeartSliceBuilder()
heart_slice_builder.build_from_file(path, heart, slice_name,
                                    n_angular, n_radial,
                                    node_step)
heart_slice = heart_slice_builder.heart_slice

map, segments = SquaredSegments.random_segments(heart_slice.image,
                                                heart_slice.wall_mask,
                                                size=200,
                                                number_of_segments=30)

colleclted_props = []
for segment in tqdm(segments):
    props = PatternPropertiesBuilder().build(segment == 2)
    colleclted_props.append(props)

props = pd.concat(colleclted_props, ignore_index=True)

print(props.describe())

res = heart_slice.wall_mask.copy().astype(int)
res += map

fig, axs = plt.subplots(ncols=3)
axs[0].imshow(res, cmap='viridis')
axs[0].axis('off')

axs[1].hist(props['structural_anisotropy'], bins=np.arange(1, 10))
axs[1].set_title('Structural Anisotropy')

axs[2].hist(props['complexity'], bins=20)
axs[2].set_title('Complexity')

plt.show()


sa = props['structural_anisotropy'].values
cm = props['complexity'].values
idx_min = np.argmin(sa)
idx_max = np.argmax(sa)

idx_mean = np.abs(sa - np.mean(sa)).argmin()

idx_25 = np.abs(sa - np.quantile(sa, 0.25)).argmin()
idx_50 = np.abs(sa - np.quantile(sa, 0.5)).argmin()
idx_75 = np.abs(sa - np.quantile(sa, 0.75)).argmin()

print(np.quantile(sa, 0.25), np.quantile(sa, 0.5), np.quantile(sa, 0.75))
print(idx_min, idx_max, idx_mean, idx_25, idx_50, idx_75)

fig, axs = plt.subplots(ncols=3, nrows=2)
axs = axs.ravel()

axs[0].imshow(segments[idx_min], cmap=cmap, vmin=0, vmax=2)
axs[0].set_title(f'Min: {sa[idx_min]:.2f}, {cm[idx_min]:.2f}')
axs[1].imshow(segments[idx_mean], cmap=cmap, vmin=0, vmax=2)
axs[1].set_title(f'Mean: {sa[idx_mean]:.2f}, {cm[idx_mean]:.2f}')
axs[2].imshow(segments[idx_max], cmap=cmap, vmin=0, vmax=2)
axs[2].set_title(f'Max: {sa[idx_max]:.2f}, {cm[idx_max]:.2f}')
axs[3].imshow(segments[idx_25], cmap=cmap, vmin=0, vmax=2)
axs[3].set_title(f'25%: {sa[idx_25]:.2f}, {cm[idx_25]:.2f}')
axs[4].imshow(segments[idx_50], cmap=cmap, vmin=0, vmax=2)
axs[4].set_title(f'50%: {sa[idx_50]:.2f}, {cm[idx_50]:.2f}')
axs[5].imshow(segments[idx_75], cmap=cmap, vmin=0, vmax=2)
axs[5].set_title(f'75%: {sa[idx_75]:.2f}, {cm[idx_75]:.2f}')

for ax in axs:
    ax.axis('off')

plt.tight_layout()
plt.show()
