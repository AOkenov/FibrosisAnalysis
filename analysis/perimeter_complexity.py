from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import morphology, segmentation, measure, filters
from scipy import ndimage
from sklearn import neighbors, ensemble

from fibrosisanalysis.parsers import ImageLoader, StatsLoader
from fibrosisanalysis.analysis import ObjectsPropertiesBuilder
from fibrosisanalysis.plots.point_density import PointDensity
from fibrosisanalysis.morphology.erosion_segmentation import ErosionSegmentation


path = Path(__file__).parents[1].joinpath('data')

heart = 'E11444_LMNA'
slice_name = 'E11444_08_SC2'

# heart = 'E10927_MYBPC3'
# slice_name = 'E10927_08_SC2'


def compute_outliers(complexity):
    iqr = np.percentile(complexity, 75) - np.percentile(complexity, 25)
    outliers = complexity > np.percentile(complexity, 75) + 1.5 * iqr
    return outliers


image_loader = ImageLoader(path)
image = image_loader.load_slice_data(path.joinpath(heart, 'Images',
                                                   slice_name))

# labeled = ErosionSegmentation.segment(image > 1)

labeled = measure.label(image > 1, connectivity=1)
props = measure.regionprops_table(labeled, properties=('label', 'area',
                                                       'solidity', 'image',
                                                       'perimeter_crofton',
                                                       'area_convex'))
props['complexity'] = props['perimeter_crofton'] ** 2 / (4 * np.pi * props['area_convex'])

props = pd.DataFrame(props)

props['complexity'].fillna(1, inplace=True)

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

df = props[props['area'] > 10]
area = df['area'].values
complexity_val = df['complexity'].values
complexity_val_a, area, adens = PointDensity.sort_by_density(complexity_val,
                                                             area)

solidity = df['solidity'].values

complexity_val_s, solidity, sdens = PointDensity.sort_by_density(complexity_val,
                                                                 solidity)


# classifier = neighbors.LocalOutlierFactor(n_neighbors=20)
classifier = ensemble.IsolationForest(contamination=0.05)
outliers = classifier.fit_predict(np.array([complexity_val_s, solidity]).T)

fig = plt.figure()
axs = fig.subplot_mosaic([['solidity', 's_label'],
                          ['area', 'area_label'],],
                         sharex=True, sharey=True)
axs['solidity'].scatter(solidity, complexity_val_s, c=sdens)
axs['s_label'].scatter(solidity, complexity_val_s, c=outliers)

axs['area'].scatter(area, complexity_val_a, c=adens)
axs['area_label'].scatter(area, complexity_val_a, c=outliers)
plt.show()

# area_mask = props['area'].between(5000, 6300)
# area_mask = props['complexity'] > 10

outliers = classifier.predict(props[['complexity', 'solidity']].values)

area_mask = outliers.flatten() == -1

area_mask &= props['area'].values > 10
area_mask_map = area_mask[labeled - 1]
area_mask_map[labeled == 0] = 0

complexity_map = props['complexity'].values[labeled - 1]
complexity_map[area_mask_map == 0] = 0
# complexity_map[complexity_map > 0] = np.log(complexity_map[complexity_map > 0])

solidity_map = props['solidity'].values[labeled - 1]
solidity_map = 1 - solidity_map
solidity_map[area_mask_map == 0] = 0

area_map = props['area'].values[labeled - 1]
area_map[area_mask_map == 0] = 0
# area_map[area_map > 0] = np.log(area_map[area_map > 0])

fig = plt.figure()
axs = fig.subplot_mosaic([['image', 'area'],
                          ['solidity', 'complexity']],
                         sharex=True, sharey=True)

axs['image'].imshow(image, cmap='viridis', origin='lower')
axs['image'].set_title('Image')

axs['solidity'].imshow(solidity_map, cmap='viridis', origin='lower')
axs['solidity'].set_title('Solidity')

axs['area'].imshow(area_map, cmap='viridis', origin='lower')
axs['area'].set_title('Area')

axs['complexity'].imshow(complexity_map, cmap='viridis', origin='lower')
axs['complexity'].set_title('Complexity')
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
