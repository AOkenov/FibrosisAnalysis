from fibrosisprocessing.plots import Data

import numpy as np


path = '/home/venus/Projects/heartrecon/wetransfer-21bce0/'

hearts = ['E10615_MYH7', 'E10621_ABCC9', 'E10691_RBM20', 'E10788_LMNA',
          'E10884', 'E10927_MYBPC3', 'E11442_TTN', 'E11443_LMNA',
          'E11444_LMNA', 'E11971_MYH7']


 def load(paths):
    for path in paths:
        for file in os.listdir(path):
            df.append(pd.read_pickle(path + file))

    df = pd.concat(df, ignore_index=True)


paths = []
for heart in hearts:
    paths.append(path + "{}/Stats/".format(heart))

data = Data()
data.load(paths)

mask = data.mask('area', 10, np.inf)
df = data.df[mask]

df.to_pickle(path + 'data_10.csv')
