from pathlib import Path
import warnings
import pandas as pd
from tqdm import tqdm


class Loader:
    def __init__(self, path, subdir='', file_type=''):
        """
        Initialize the Loader object.

        Parameters
        ----------
        path : str
            The base path where the data files are located.
        subdir : str, optional
            The subdirectory within the base path. Defaults to ''.
        file_type : str, optional
            The file extension/type to filter the files. Defaults to ''.
        """
        self.path = Path(path)
        self.subdir = subdir
        self.file_type = file_type

    def load_slice_data(self, path):
        """
        Load the slice data from a file.

        Parameters
        ----------
        path : str
            The path to the file.

        Raises
        ------
        NotImplementedError
            This method is not implemented.

        Returns
        -------
        pd.DataFrame
            The loaded slice data.
        """
        raise NotImplementedError('load_slice_data not implemented')

    def load_heart_data(self, heart, pbar_description='1/1'):
        """
        Load the heart data for a specific heart.

        Parameters
        ----------
        heart : str
            The name of the heart.
        pbar_description : str, optional
            The description for the progress bar. Defaults to '1/1'.

        Returns
        -------
        pd.DataFrame
            The loaded heart data.
        """
        path = self.path.joinpath(heart, self.subdir)
        files = list(path.glob('*{}'.format(self.file_type)))
        files = [file for file in files if not file.name.startswith('.')]

        data = []
        with tqdm(total=len(files)) as pbar:
            pbar.set_description(pbar_description)
            for file in files:
                slice_data = self.load_slice_data(file, asdf=True)
                slice_data['Slice'] = file.stem
                data.append(slice_data)
                pbar.update()

        if len(data) == 0:
            warnings.warn('No data found for heart {}'.format(heart))
            return pd.DataFrame()

        return pd.concat(data, ignore_index=True)

    def load_hearts_data(self, hearts):
        """
        Load the heart data for multiple hearts.

        Parameters
        ----------
        hearts : list
            A list of heart names.

        Returns
        -------
        pd.DataFrame
            The loaded heart data for all hearts.
        """
        data = []
        for i, heart in enumerate(hearts):
            pbar_description = '{}/{}'.format(i + 1, len(hearts))
            heart_data = self.load_heart_data(heart, pbar_description)
            heart_data['Heart'] = heart
            data.append(heart_data)
        return pd.concat(data, ignore_index=True)
