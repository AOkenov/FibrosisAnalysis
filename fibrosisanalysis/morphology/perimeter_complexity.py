import numpy as np
from skimage import measure


class PerimeterComplexity:
    def __init__(self):
        pass

    @staticmethod
    def image_complexity(image):
        """
        Calculate the perimeter complexity of a binary image as
        the ratio of the perimeter square to the area.

        Parameters
        ----------
        image : np.ndarray
            Binary image.

        Returns
        -------
        float
            Perimeter complexity.
        """
        area = np.sum(image > 0)
        perimeter = measure.perimeter_crofton(image > 0, directions=4)
        complexity = perimeter ** 2 / (4 * np.pi * area)
        return complexity

    @staticmethod
    def objects_complexity(labeled):
        """
        Calculate the perimeter complexity of each object in a labeled image.

        Parameters
        ----------
        labeled : np.ndarray
            Labeled image.

        Returns
        -------
        np.ndarray
            Array with the perimeter complexity of each object.
        """
        complexity = []
        for label in np.arange(1, labeled.max() + 1):
            mask = labeled == label
            complexity.append(PerimeterComplexity.image_complexity(mask))
        return np.array(complexity)
