from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, segmentation, measure, filters
from scipy import ndimage

from fibrosisanalysis.parsers import ImageLoader
from fibrosisanalysis.analysis import ObjectsPropertiesBuilder


path = Path(__file__).parents[1].joinpath('data_gen')

original_images = [f.stem for f in path.joinpath('original_texs').glob('*.png')]


def compute_props(labeled):
    perimeter = []
    solitidy = []
    area = []
    for label in np.arange(1, labeled.max() + 1):
        mask = labeled == label
        perimeter.append(measure.perimeter_crofton(mask, directions=4))
        solitidy.append(measure.regionprops(mask.astype(int))[0].solidity)
        area.append(mask.sum())

    perimeter = np.array(perimeter)
    solitidy = np.array(solitidy)
    area = np.array(area)

    complexity = perimeter ** 2 / (4 * np.pi * area)
    return complexity, solitidy


def compute_outliers(complexity):
    iqr = np.percentile(complexity, 75) - np.percentile(complexity, 25)
    outliers = complexity > np.percentile(complexity, 75) + 1.5 * iqr
    return outliers


for f in original_images[:5]:
    image_loader = ImageLoader()
    original = image_loader.load_slice_data(path.joinpath('original_texs', f))
    replicated = np.load(path.joinpath('sim_dir_2', f).with_suffix('.npy'))

    # binary = replicated > 1
    binary = original > 1
    binary = morphology.remove_small_objects(binary, min_size=10)
    # binary = segmentation.clear_border(binary)
    labeled, n_labels = measure.label(binary, connectivity=1, return_num=True)

    labels = np.arange(1, labeled.max() + 1)

    # image = np.zeros((3, 3))
    # image[1:3, 1:3] = 1
    # # image[2:, 2:8] = 0

    # print(measure.perimeter_crofton(image, directions=4))
    # print(measure.perimeter(image))

    complexity, solitidy = compute_props(labeled)

    # complexity_image = complexity[labeled - 1]
    # complexity_image[labeled == 0] = 0
    # complexity_mask = complexity_image > 10

    outliers = compute_outliers(complexity)

    complexity_mask = np.isin(labeled, labels[outliers])

    # plt.figure()
    # plt.imshow(complexity_mask, cmap='viridis')
    # plt.title('Complexity mask')
    # plt.show()

    distance = ndimage.distance_transform_edt(complexity_mask)

    # centroid_mask = morphology.binary_erosion(complexity_mask)
    centroid_mask = distance > 1
    centroid_mask = morphology.remove_small_objects(centroid_mask, min_size=10)
    centroid_values = ndimage.distance_transform_edt(centroid_mask)

    complexity_labeled, pos = measure.label(centroid_mask, connectivity=1,
                                            return_num=True)

    centroids = ndimage.minimum_position(-centroid_values, complexity_labeled,
                                        index=np.arange(1, pos+1))

    blured = centroid_mask.copy().astype(int)

    coords = np.array(centroids)

    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    blured = segmentation.watershed(-distance, markers, mask=complexity_mask,
                                    watershed_line=True)

    binary[complexity_mask] = 0
    labeled_old = measure.label(binary, connectivity=1)

    # plt.figure()
    # plt.imshow(labeled_old, cmap='viridis')
    # plt.title('Labeled old')
    # plt.show()

    blured[~complexity_mask] = blured.max() + labeled_old[~complexity_mask]
    blured[labeled == 0] = 0

    labels_2 = np.arange(1, blured.max() + 1)
    complexity_2, solitidy_2 = compute_props(blured)

    outliers_2 = compute_outliers(complexity_2)
    print(labels_2[outliers_2])

    # labeled[complexity_mask] = n_labels + blured[complexity_mask]
    # solitidy = np.concatenate((solitidy, solitidy_2))
    # complexity = np.concatenate((complexity, complexity_2))

    # complexity_2_image = complexity_2[blured - 1]
    # complexity_2_mask = complexity_2_image > 5

    # blured[complexity_2_mask < 1] = 0

    fig, axs = plt.subplots(ncols=2, nrows=2)

    axs[1, 1].sharex(axs[1, 0])
    axs[1, 1].sharey(axs[1, 0])

    axs[0, 1].sharex(axs[0, 0])

    axs[0, 0].plot(solitidy, complexity, 'o')
    axs[0, 0].plot(solitidy[outliers], complexity[outliers], 'o', color='red')
    axs[0, 1].plot(solitidy_2, complexity_2, 'o')
    axs[0, 1].plot(solitidy_2[outliers_2], complexity_2[outliers_2], 'o', color='red')

    for label, x, y in zip(np.arange(1, labeled.max() + 1), solitidy, complexity):
        axs[0, 0].text(x, y, label)

    for label, x, y in zip(labels_2, solitidy_2, complexity_2):
        axs[0, 1].text(x, y, label, color='red')

    labeled = np.ma.masked_where(labeled == 0, labeled)
    axs[1, 0].imshow(labeled, cmap='viridis')

    blured = np.ma.masked_where(blured == 0, blured)
    axs[1, 1].imshow(blured, cmap='viridis')
    plt.show()
