from pathlib import Path
import os
import numpy as np
from tqdm import tqdm

from fibrosisanalysis.parsers import StatsLoader


def copy_files_with_pattern(src_dir):
    # Iterate over files in source directory
    for filename in src_dir:
        pattern = ''
        new_pattern = ''
        if '_WABL' in filename.stem:
            pattern = '_WABL'
            new_pattern = '_NABL'

        dest_file = filename.parent.joinpath(filename.stem.replace(pattern,
                                                                   new_pattern))
        filename.rename(dest_file.with_suffix('.pkl'))


path = Path(__file__).parent.parent.joinpath('data')
# path = save_path

hearts = ['E10615_MYH7', 'E10621_ABCC9', 'E10691_RBM20', 'E10788_LMNA',
          'E10884', 'E10927_MYBPC3', 'E11442_TTN', 'E11443_LMNA',
          'E11444_LMNA', 'E11971_MYH7']

for heart in hearts[1:]:
    slice_names = list(path.joinpath(heart, 'StatsGenerated').glob('*'))
    # copy_files_with_pattern(slice_names)

import pandas as pd

# print(pd.read_pickle(slice_names[0]))
