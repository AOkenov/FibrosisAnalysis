from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import morphology, segmentation, measure, filters
from scipy import ndimage
from sklearn import neighbors, ensemble

from fibrosisanalysis.parsers import ImageLoader, StatsLoader, DensityLoader
from fibrosisanalysis.analysis import ObjectsPropertiesBuilder
from fibrosisanalysis.plots.point_density import PointDensity
from fibrosisanalysis.morphology.erosion_segmentation import ErosionSegmentation


def fibrosis_density(image, intensity):
    return intensity[image > 0].mean()


# path = Path(__file__).parents[1].joinpath('data')
path = Path('/Users/arstanbek/Library/CloudStorage/OneDrive-UGent/data')

heart = 'E11444_LMNA'
slice_name = 'E11444_08_SC2'

# heart = 'E10927_MYBPC3'
# slice_name = 'E10927_08_SC2'

image_loader = ImageLoader(path)
image = image_loader.load_slice_data(path.joinpath(heart, 'Images',
                                                   slice_name))
density_loader = DensityLoader(path)
density_map = density_loader.load_slice_data(path.joinpath(heart, 'Density',
                                                           slice_name))

# labeled = ErosionSegmentation.segment(image > 1)
labeled = measure.label(image > 1, connectivity=1)

props = measure.regionprops_table(labeled, intensity_image=density_map,
                                  properties=('label', 'area', 'solidity',
                                              'perimeter_crofton',
                                              'area_convex'),
                                  extra_properties=(fibrosis_density,))

props['complexity'] = props['perimeter_crofton'] ** 2 / (4 * np.pi
                                                         * props['area'])

props = pd.DataFrame(props)

props['complexity'].fillna(1, inplace=True)

df = props[(props['area'] > 10) & (props['fibrosis_density'] > 0.2) & (props['fibrosis_density'] < 0.3)]
area = df['area'].values
complexity = df['complexity'].values
solidity = df['solidity'].values
fibrosis = df['fibrosis_density'].values

complexity_a, area_a, dens_a = PointDensity.sort_by_density(complexity, area)
complexity_s, solidity_s, dens_s = PointDensity.sort_by_density(complexity,
                                                                solidity)
complexity_f, fibrosis_f, dens_f = PointDensity.sort_by_density(complexity,
                                                                fibrosis)


# classifier = neighbors.LocalOutlierFactor(n_neighbors=20)
classifier = ensemble.IsolationForest(contamination=0.1)
outliers = classifier.fit_predict(df[['complexity',
                                      'solidity']].values)

fig = plt.figure()
axs = fig.subplot_mosaic([['solidity', 'density', 'area'],
                          ['solidity_label', 'density_label', 'area_label']],)
axs['solidity'].scatter(solidity_s, complexity_s, c=dens_s)
axs['solidity_label'].scatter(solidity, complexity, c=outliers)
axs['solidity'].sharex(axs['solidity_label'])
axs['solidity'].sharey(axs['solidity_label'])
axs['solidity'].set_ylabel('Complexity')
axs['solidity'].set_xlabel('Solidity')

axs['density'].scatter(fibrosis_f, complexity_f, c=dens_f)
axs['density_label'].scatter(fibrosis, complexity, c=outliers)
axs['density'].sharex(axs['density_label'])
axs['density'].sharey(axs['density_label'])
axs['density'].set_ylabel('Complexity')
axs['density'].set_xlabel('Density')

axs['area'].scatter(area_a, complexity_a, c=dens_a)
axs['area'].set_xscale('log')
axs['area_label'].scatter(area, complexity, c=outliers)
axs['area_label'].set_xscale('log')
axs['area'].sharex(axs['area_label'])
axs['area'].sharey(axs['area_label'])
axs['area'].set_ylabel('Complexity')
axs['area'].set_xlabel('Area')
plt.show()

# For full map
outliers = classifier.predict(props[['complexity',
                                     'solidity']].values)

outliers_mask = outliers.flatten() == -1
outliers_mask &= props['area'].values > 10
outliers_map = outliers_mask[labeled - 1]
outliers_map[labeled == 0] = 0

complexity_map = props['complexity'].values[labeled - 1]
complexity_map[labeled == 0] = 0
# complexity_map[outliers_map == 0] = 0
# complexity_map[complexity_map > 0] = np.log(complexity_map[complexity_map > 0])

solidity_map = props['solidity'].values[labeled - 1]
solidity_map = 1 - solidity_map
solidity_map[labeled == 0] = 0
# solidity_map[outliers_map == 0] = 0

area_map = props['area'].values[labeled - 1]
area_map[labeled == 0] = 0
# area_map[outliers_map == 0] = 0
# area_map[area_map > 0] = np.log(area_map[area_map > 0])

density_map = props['fibrosis_density'].values[labeled - 1]
density_map[labeled == 0] = 0

fig = plt.figure()
axs = fig.subplot_mosaic([['image', 'density'],
                          ['solidity', 'complexity']],
                         sharex=True, sharey=True)

axs['image'].imshow(image + outliers_map, cmap='viridis', origin='lower')
axs['image'].set_title('Image')

axs['solidity'].imshow(solidity_map, cmap='viridis', origin='lower')
axs['solidity'].set_title('Solidity')

axs['density'].imshow(density_map, cmap='viridis', origin='lower')
axs['density'].set_title('Density')

axs['complexity'].imshow(complexity_map, cmap='viridis', origin='lower')
axs['complexity'].set_title('Complexity')
plt.show()
