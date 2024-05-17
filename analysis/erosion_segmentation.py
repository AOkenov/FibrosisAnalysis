from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from skimage import measure, morphology
from tqdm import tqdm
from natsort import natsorted

from fibrosisanalysis.parsers import ImageLoader
from fibrosisanalysis.morphology.perimeter_complexity import PerimeterComplexity
from fibrosisanalysis.morphology.erosion_segmentation import ErosionSegmentation

path = Path(__file__).parents[1].joinpath('data_gen')

files = [f.stem for f in path.joinpath('original_texs').glob('*.png')]
files = natsorted([f for f in files if '200' in f])
filename = files[0]
collected_props = []
collected_props_eroded = []
for filename in tqdm(files[:1]):
    image_loader = ImageLoader()
    image = image_loader.load_slice_data(path.joinpath('original_texs', filename))
    mask = image > 1
    mask = morphology.remove_small_objects(mask, min_size=10)

    labeled = measure.label(mask, connectivity=1)

    props_list = ['label', 'solidity', 'area', 'perimeter_crofton', 'area_convex']

    props = measure.regionprops_table(labeled, properties=props_list)
    props['complexity'] = props['perimeter_crofton'] ** 2 / (4 * np.pi
                                                             * props['area_convex'])
    props['filename'] = filename
    props = pd.DataFrame(props)
    collected_props.append(props)

    eroded_labels = ErosionSegmentation().segment(mask)
    eroded_props = measure.regionprops_table(eroded_labels, properties=props_list)
    eroded_props['complexity'] = (eroded_props['perimeter_crofton'] ** 2
                                  / (4 * np.pi * eroded_props['area_convex']))
    eroded_props['filename'] = filename
    eroded_props = pd.DataFrame(eroded_props)
    collected_props_eroded.append(eroded_props)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(labeled, cmap='viridis')
    axs[0].set_title('Original')
    axs[1].imshow(eroded_labels, cmap='viridis')
    axs[1].set_title('Eroded')
    plt.show()

    plt.figure()
    plt.scatter(props['solidity'], props['complexity'])
    plt.scatter(eroded_props['solidity'], eroded_props['complexity'])
    plt.xlabel('Solidity')
    plt.ylabel('Complexity')
    # plt.yscale('log')
    plt.show()

# collected_props = pd.concat(collected_props)
# collected_props_eroded = pd.concat(collected_props_eroded)

# complexity_sum = collected_props.groupby('filename')['complexity'].sum()
# print(complexity_sum)

# complexity_sum_eroded = collected_props_eroded.groupby('filename')['complexity'].sum()
# print(complexity_sum_eroded)

# plt.figure()
# plt.plot(complexity_sum)
# plt.show()

# plt.figure()
# plt.plot(complexity_sum_eroded)
# plt.xticks(rotation=90)
# plt.show()