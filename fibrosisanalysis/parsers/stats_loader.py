from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from fibrosisanalysis.parsers.loader import Loader


class StatsLoader(Loader):
    """StatsLoader class for loading and processing data from CSV files
    into DataFrames.

    Methods
    -------
    load_slice_stats(path, file):
        Load a single slice of data from a CSV file.

    load_heart_stats(path, heart, stats_folder='Stats'):
        Load data for a specific heart from CSV files in the specified
        stats folder.

    load_hearts_stats(path, hearts, stats_folder='Stats'):
        Load data for multiple hearts from CSV files in the specified
        stats folder.

    setup_data(df):
        Perform data setup and feature engineering on the input DataFrame.

    Attributes
    ----------
    None
    """

    def __init__(self, path='.', subdir='Stats', collected_columns=None):
        """Initialize an instance of StatsLoader.
        """
        super().__init__(path, subdir=subdir, file_type='.pkl')
        self.collected_columns = collected_columns

    def load_slice_data(self, path, asdf=True):
        """Load a single slice of data from a pkl file.

        Parameters
        ----------
        path : str
            Path to the directory containing the pkl file.
        file : str
            Name of the pkl file (with or without extension).

        Returns
        -------
        pd.DataFrame
            Loaded data as a Pandas DataFrame.
        """
        path = Path(path)
        data = pd.read_pickle(path.with_suffix(self.file_type))
        data['FileName'] = path.stem

        if self.collected_columns is not None:
            data = data[self.collected_columns]

        # data = pd.read_csv(path.with_suffix('.csv'), usecols=columns)
        # data.to_csv(path.joinpath(file).with_suffix('.csv'))
        return data
