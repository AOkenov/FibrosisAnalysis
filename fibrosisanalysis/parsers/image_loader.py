from pathlib import Path
import numpy as np
import pandas as pd
from skimage import io

from fibrosisanalysis.parsers.loader import Loader


class ImageLoader(Loader):
    """
    A class that provides methods for loading and processing images.
    """

    def __init__(self, path=''):
        super().__init__(path, subdir='Images', file_type='.png')
    
    def load_png(self, path):
        """
        Load a PNG image from the specified path and file.

        Parameters
        ----------
        path : str
            The path to the directory containing the image file.

        Returns
        -------
        numpy.ndarray
            The loaded image as a NumPy array.
        """
        path = Path(path)
        array = io.imread(path.with_suffix(self.file_type), as_gray=True)
        return array

    def rescale(self, image):
        """
        Rescale the pixel values of the image.

        Parameters
        ----------
        image : numpy.ndarray
            The input image.

        Returns
        -------
        numpy.ndarray
            The rescaled image.
        """
        return np.add(image < 0.4, image < 0.8, dtype=int)
    
    def load_slice_data(self, path, asdf=False, rescale=True):
        """
        Load a single image slice from the specified path and file.

        Parameters
        ----------
        path : str
            The path to the directory containing the image file.
        asdf : bool, optional
            Whether to return the image as a DataFrame, by default False.
        rescale : bool, optional
            Whether to rescale the image, by default True.

        Returns
        -------
        numpy.ndarray
            The loaded and optionally rescaled image slice.
        """
        path = Path(path)
        image = self.load_png(path)

        if not rescale:
            return image
        
        image = self.rescale(image)

        if not asdf:
            return image
        
        data = {'Image': [image]}
        return pd.DataFrame(data)
