import numpy as np


class SquaredSegments:
    def __init__(self) -> None:
        pass

    @staticmethod
    def random_segments(image, mask, size=100, number_of_segments=10):
        """Generate random segments on the image.

        Parameters
        ----------
        image : np.ndarray
            Image on which to draw random segments.
        mask : np.ndarray
            Mask on which to draw random segments.
        size : int, optional
            Size of the segments, by default 100.
        number_of_segments : int, optional
            Number of segments to draw, by default 10.

        Returns
        -------
        list
            List of random segments.
        """

        coords = np.argwhere(mask > 0)
        segments = []

        out = np.zeros_like(mask)
        
        i = 1
        while len(segments) < number_of_segments:
            idx = np.random.randint(0, len(coords))
            x, y = coords[idx]

            segment = image[x:x + size, y:y + size]

            if (np.sum(segment == 2) / size ** 2) > 0.1:
                continue

            coverage = np.sum(out[x:x + size, y:y + size] > 0) / size ** 2

            if coverage > 0.3:
                continue

            if mask[x:x + size, y:y + size].all():    
                segment[segment == 0] = 1
                segments.append(segment)
                out[x:x + size, y:y + size] = i
                i += 1

        return out, segments