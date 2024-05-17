from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import morphology, segmentation, measure, filters
from scipy import ndimage

from fibrosisanalysis.parsers import ImageLoader, StatsLoader
from fibrosisanalysis.analysis import ObjectsPropertiesBuilder
from fibrosisanalysis.plots.point_density import PointDensity


path = Path('/Users/arstanbek/Library/CloudStorage/OneDrive-UGent/data')

path = Path('/Users/arstanbek/Library/CloudStorage/OneDrive-UGent/data')

heart = 'E11444_LMNA'
slice_name = 'E11444_08_SC2'


def complexity(image):
    area = np.sum(image > 0)
    perimeter = measure.perimeter_crofton(image > 0, directions=4)
    complexity = perimeter ** 2 / (4 * np.pi * area)
    return complexity


def compute_outliers(complexity):
    iqr = np.percentile(complexity, 75) - np.percentile(complexity, 25)
    outliers = complexity > np.percentile(complexity, 75) + 1.5 * iqr
    return outliers


image_loader = ImageLoader(path)
image = image_loader.load_slice_data(path.joinpath(heart, 'Images',
                                                   slice_name))

labeled = measure.label(image > 1, connectivity=1)
props = measure.regionprops_table(labeled, properties=('label', 'area',
                                                       'solidity', 'image'),
                                  extra_properties=(complexity,))

props = pd.DataFrame(props)

complexity_map = props['complexity'].values[labeled - 1]
complexity_map[labeled == 0] = 0

props = props[props['area'] > 10]
print(props.head())

df = props[props['area'].between(1000, 1300)]

inds = np.argsort(-df['complexity'].values)

print(df['area'].iloc[inds[0]], df['area'].iloc[inds[-1]])

print(inds.shape)

fig, axs = plt.subplots(ncols=2, nrows=1)
axs[0].imshow(df['image'].iloc[inds[0]], cmap='gray', origin='lower')
axs[0].set_title('Complexity: {:.2f} \n Area: {:.0f} \n Solidity: {:.2f}'.format(df['complexity'].iloc[inds[0]],
                                                                                 df['area'].iloc[inds[0]],
                                                                                 df['solidity'].iloc[inds[0]]))
axs[1].imshow(df['image'].iloc[inds[-1]], cmap='gray', origin='lower')
axs[1].set_title('Complexity: {:.2f} \n Area: {:.0f} \n Solidity: {:.2f}'.format(df['complexity'].iloc[inds[-1]],
                                                                                 df['area'].iloc[inds[-1]],
                                                                                 df['solidity'].iloc[inds[-1]]))
plt.show()

complexity_val = np.log10(props['complexity'].values)
area = np.log10(props['area'].values)
complexity_val_a, area, adens = PointDensity.sort_by_density(complexity_val,
                                                             area)

solidity = props['solidity'].values

complexity_val_s, solidity, sdens = PointDensity.sort_by_density(complexity_val,
                                                                 solidity)


fig, axs = plt.subplots(ncols=2, nrows=1)
axs[0].scatter(solidity, complexity_val_s, c=sdens)
axs[1].scatter(area, complexity_val_a, c=adens)
plt.show()

plt.figure()
plt.imshow(complexity_map, cmap='viridis', origin='lower')
plt.show()


# df = df[df['area'] > 10]
# df['perimeter'] = df['image'].apply(compute_props)

# complexity = df['perimeter'] ** 2 / (4 * np.pi * df['area'])
# outliers = compute_outliers(complexity.values)
# solitidy = df['solidity']

# max_inds = np.argsort(-complexity.values)

# print(np.count_nonzero(outliers))

# min_outlier_idx = max_inds[550]

# # print(min_outlier_idx)

# fig, axs = plt.subplots(ncols=2, nrows=1)

# axs[0].plot(solitidy, complexity, 'o')
# axs[0].plot(solitidy[outliers], complexity[outliers], 'o', color='red')

# axs[1].imshow(df['image'].iloc[min_outlier_idx], cmap='gray', origin='lower')
# plt.show()
