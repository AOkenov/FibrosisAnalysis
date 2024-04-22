import os
import numpy as np
import pandas as pd
from skimage import io


class IO:
    def __init__(self):
        pass

    @staticmethod
    def load_png(path, dirname, file):
        array = io.imread(path + dirname + file + '.png', as_gray=True)
        return array

    @staticmethod
    def load_rescaled_png(path, file, dirname='Images/', ablation='NABL'):
        file = file.replace('WABL', ablation)
        array = IO.load_png(path, dirname, file)
        return np.add(array < 0.4, array < 0.8, dtype=int)

    @staticmethod
    def load_edge(path, filename, suffix=''):
        return np.load(path + 'Edges/' + filename + suffix + '.npy')

    @staticmethod
    def save_png(path, dirname, file, array):
        if not os.path.isdir(path + dirname):
            os.mkdir(path + dirname)

        io.imsave(path + dirname + '{}.png'.format(file),
                  array.astype(np.uint8))

    @staticmethod
    def load_df(path, dirname, file):
        return pd.read_pickle(path + dirname + file + '.csv')

    @staticmethod
    def load_df_from_list(path, dirname, files):
        df = []
        for file in files:
            df.append(pd.read_pickle(path + dirname + file + '.csv'))

        return pd.concat(df, ignore_index=True)

    @staticmethod
    def save_df(path, dirname, file, df):
        if not os.path.exists(path + dirname):
            os.mkdir(path + dirname)
        df.to_pickle(path + dirname + file + '.csv')
