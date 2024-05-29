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
# path = Path('/Users/arstanbek/Library/CloudStorage/OneDrive-UGent/data')

heart = 'E11444_LMNA'
slice_name = 'E11444_08_SC2'

# heart = 'E10927_MYBPC3'
# slice_name = 'E10927_08_SC2'

image_loader = ImageLoader(path)
image = image_loader.load_slice_data(path.joinpath(heart, 'Images',
                                                   slice_name))

# labeled = ErosionSegmentation.segment(image > 1)

labeled = measure.label(image > 1, connectivity=1)
props = measure.regionprops_table(labeled, properties=('label', 'area',
                                                       'solidity', 'image',
                                                       'perimeter_crofton',
                                                       'area_convex'))
props['complexity'] = (props['perimeter_crofton'] ** 2
                       / (4 * np.pi * props['area']))

props = pd.DataFrame(props)
props.fillna({'complexity': 1}, inplace=True)

print(props.head())

df = props[props['area'].between(1000, 1300)]

inds = np.argsort(-df['complexity'].values)

print(df['area'].iloc[inds[0]], df['area'].iloc[inds[-1]])

print(inds.shape)

fig, axs = plt.subplots(ncols=2, nrows=1)
axs[0].imshow(df['image'].iloc[inds[0]], cmap='gray', origin='lower')
axs[0].set_title('Complexity: {:.2f} \n'.format(df['complexity'].iloc[inds[0]])
                 + 'Area: {:.0f} \n'.format(df['area'].iloc[inds[0]])
                 + 'Solidity: {:.2f}'.format(df['solidity'].iloc[inds[0]]))

axs[1].imshow(df['image'].iloc[inds[-1]], cmap='gray', origin='lower')
axs[1].set_title('Complexity: {:.2f} \n'.format(df['complexity'].iloc[inds[-1]])
                 + 'Area: {:.0f} \n'.format(df['area'].iloc[inds[-1]])
                 + 'Solidity: {:.2f}'.format(df['solidity'].iloc[inds[-1]]))
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
                         sharey=True)
axs['solidity'].scatter(solidity, complexity_val_s, c=sdens)
axs['s_label'].scatter(solidity, complexity_val_s, c=outliers)

axs['area'].scatter(area, complexity_val_a, c=adens)
axs['area_label'].scatter(area, complexity_val_a, c=outliers)
plt.show()

# area_mask = props['area'].between(5000, 6300)
# area_mask = props['complexity'] > 10

outliers = classifier.predict(props[['complexity', 'solidity']].values)

area_mask = outliers.flatten() > -2

area_mask &= (props['area'].values > 1000) & (props['area'].values < 1300)

area_mask_map = area_mask[labeled - 1]
area_mask_map[labeled == 0] = 0

outliers_map = outliers[labeled - 1]
outliers_map[area_mask_map == 0] = 0

eroded_mask = ErosionSegmentation.segment(area_mask_map)

for i, im in enumerate(props['image'][area_mask].values):

    eroded = ErosionSegmentation.segment(im > 0)
    eroded_label = measure.label(eroded, connectivity=1)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(im, cmap='viridis', origin='lower')
    axs[0].set_title('Original')
    axs[1].imshow(eroded_label, cmap='viridis', origin='lower')
    axs[1].set_title('Segmented')
    plt.show()

    fig.savefig(f'segmented_{i}.png', dpi=300)

complexity_map = props['complexity'].values[labeled - 1]
complexity_map[area_mask_map == 0] = 0
# complexity_map[complexity_map > 0] = np.log(complexity_map[complexity_map > 0])

solidity_map = props['solidity'].values[labeled - 1]
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

axs['solidity'].imshow(outliers_map, cmap='viridis', origin='lower')
axs['solidity'].set_title('Solidity')

axs['area'].imshow(eroded_mask, cmap='viridis', origin='lower')
axs['area'].set_title('Area')

axs['complexity'].imshow(complexity_map, cmap='viridis', origin='lower')
axs['complexity'].set_title('Complexity')
plt.show()
