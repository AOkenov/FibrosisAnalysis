import sys
sys.path.append('/mnt/8TBdrive/Arstan/')
sys.path.append('/mnt/8TBdrive/Arstan/Finitewave/')

from fibrosisprocessing.tools import Density
from joblib import Parallel, delayed
from tqdm import tqdm
import os


path = '/mnt/8TBdrive/Arstan/fibrosisprocessing/data/'

dirs = ['E10615_MYH7/', 'E10621_ABCC9/', 'E10691_RBM20/', 'E10788_LMNA/',
        'E10884/', 'E10927_MYBPC3/', 'E11442_TTN/', 'E11443_LMNA/',
        'E11444_LMNA/', 'E11971_MYH7/']


def compute(path, file):
    dens = Density(path, file)
    dens.compute()
    dens.save()


bar = tqdm(total=len(dirs))
for dirname in dirs[4:5]:
    files = os.listdir(path + dirname + 'Images/')
    files = [file.split('.png')[0] for file in files]
    Parallel(n_jobs=32)(delayed(compute)(path + dirname, file)
                        for file in files)
    bar.update(1)
bar.close()
