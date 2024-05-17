import numpy as np
from scipy import ndimage
from skimage import morphology, segmentation, measure


class ErosionSegmentation:
    def __init__(self) -> None:
        pass

    @staticmethod
    def segment(image, min_distance=1, min_seed_size=10):
        distance = ndimage.distance_transform_edt(image)

        seeds_mask = distance > min_distance
        seeds_mask = morphology.remove_small_objects(seeds_mask, min_seed_size)
        seeds_label, seeds_num = measure.label(seeds_mask, connectivity=1,
                                               return_num=True)
        seeds_index = np.arange(1, seeds_num + 1)

        centroids = ndimage.minimum_position(-distance, seeds_label,
                                             index=seeds_index)
        centroids = np.array(centroids)

        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(centroids.T)] = True
        markers = seeds_label.copy()
        markers[~mask] = 0

        segmented = segmentation.watershed(-distance, markers,
                                           mask=image,
                                           watershed_line=False)

        non_segmented = image.copy()
        non_segmented[segmented > 0] = 0
        non_segmented_label, n = measure.label(non_segmented, connectivity=1,
                                               return_num=True)

        non_segmented_label[segmented > 0] = n + segmented[segmented > 0]

        return non_segmented_label