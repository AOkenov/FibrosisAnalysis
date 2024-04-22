from pathlib import Path
import numpy as np
import pandas as pd

from fibrosisanalysis.parsers.loader import Loader


class EdgeLoader(Loader):
    """
    A class that provides methods for loading and processing images.
    """

    def __init__(self, path=None):
        super().__init__(path, subdir='Edges', file_type='.npy')

    def load_slice_data(self, path, asdf=False):
        """
        Load a single slice edge from the specified path.

        Parameters
        ----------
        path : str
            The path to the directory containing the image file.
        asdf : bool, optional
            Whether to return the image as a DataFrame, by default False.

        Returns
        -------
        numpy.ndarray
            The loaded and optionally rescaled image slice.
        """
        path = Path(path)
        edge = np.load(path.with_suffix(self.file_type))

        if not asdf:
            return edge

        data = {'Edges': [edge]}
        return pd.DataFrame(data)
