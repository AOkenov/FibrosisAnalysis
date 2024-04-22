import sys
sys.path.append('/mnt/8TBdrive/Arstan/')
sys.path.append('/mnt/8TBdrive/Arstan/Finitewave/')

from fibrosisprocessing.slice.slice import Slice

from tqdm import tqdm
from joblib import Parallel, delayed
import os

path = '/mnt/8TBdrive/Arstan/fibrosisprocessing/data/'

dirs = ['E10615_MYH7/', 'E10621_ABCC9/', 'E10691_RBM20/', 'E10788_LMNA/',
        'E10884/', 'E10927_MYBPC3/', 'E11442_TTN/', 'E11443_LMNA/',
        'E11444_LMNA/', 'E11971_MYH7/']


def compute(path, file):
    slc = Slice(path, file)
    slc.objects_props()
    slc.save_objects_props()


bar = tqdm(total=len(dirs))
for dirname in dirs:
    files = [f.replace('_0.npy', '') for f in os.listdir(path + dirname + 'Edges/')
             if '_0.npy' in f]
    Parallel(n_jobs=32)(delayed(compute)(path + dirname, file)
                        for file in files)
    bar.update(1)

bar.close()
