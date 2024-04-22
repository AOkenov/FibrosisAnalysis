import numpy as np
from skimage import draw, morphology, filters
from fibrosisanalysis.segmentation.distance import Distance


class RadialLines:
    """RadialLines class for drawing and labeling radial lines on an image.

    This class provides static methods to draw radial lines and label an image
    based on the closest distance to those lines.

    Methods
    -------
    draw(start, end):
        Draws a radial line between two points.

    label(image, starts, ends):
        Labels an image based on the closest radial lines to specified
        start and end points.
    """

    def __init__(self):
        """Initialize an instance of RadialLines.
        """
        pass

    @staticmethod
    def draw(start, end):
        """Draw a radial line between two points.

        Parameters
        ----------
        start : tuple
            Coordinates of the starting point (row, column).
        end : tuple
            Coordinates of the ending point (row, column).

        Returns
        -------
        np.ndarray
            Array containing the coordinates of the drawn radial line.
        """
        rr, cc = draw.line(start[0], start[1], end[0], end[1])
        return np.vstack([rr, cc]).T

    @staticmethod
    def label(image, starts, ends):
        """Label an image based on the closest radial lines to specified
        start and end points.

        Parameters
        ----------
        image : np.ndarray
            Input image to be labeled.
        starts : list of tuples
            List of starting points for radial lines.
        ends : list of tuples
            List of ending points for radial lines.

        Returns
        -------
        np.ndarray
            Labeled image based on the closest radial lines.
        """

        image_ = image.copy()

        for i, (start, end) in enumerate(zip(starts, ends)):
            line = RadialLines.draw(start, end)

            image_[line[:, 0], line[:, 1]] = 0

        image_ = morphology.remove_small_objects(image_, min_size=10)

        out, n = morphology.label(image_, connectivity=1, background=0, 
                                  return_num=True)

        line_coords = np.argwhere((image_ == 0) & (image > 0))
        label_coords = np.argwhere(out > 0)

        _, inds = Distance.shortest(label_coords, line_coords,
                                    distance_upper_bound=3)
        out[tuple(line_coords.T)] = out[tuple(label_coords[inds].T)]

        return out, n

    @staticmethod
    def label_by_distance(image, starts, ends):
        """Label an image based on the closest radial lines to specified
        start and end points.

        Parameters
        ----------
        image : np.ndarray
            Input image to be labeled.
        starts : list of tuples
            List of starting points for radial lines.
        ends : list of tuples
            List of ending points for radial lines.

        Returns
        -------
        np.ndarray
            Labeled image based on the closest radial lines.
        """
        coords = []
        labels = []

        # Draw radial lines, collect coordinates, and assign labels
        for i, (start, end) in enumerate(zip(starts, ends)):
            line = RadialLines.draw(start, end)
            coords.append(line)
            labels.append(np.ones(len(line)) * (i + 1))

        coords = np.vstack(coords)
        labels = np.hstack(labels)

        # Find the closest distance from image points to radial lines
        _, inds = Distance.shortest(coords, np.argwhere(image > 0))

        # Assign labels to the image based on the closest radial lines
        out = np.zeros_like(image, dtype=int)
        out[image > 0] = labels[inds]

        return out
