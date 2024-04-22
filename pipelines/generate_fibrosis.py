from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
import os
from fibrosisprocessing.tools import IO, Density
import sys
sys.path.append('/mnt/8TBdrive/Arstan/')
sys.path.append('/mnt/8TBdrive/Arstan/Finitewave/')


path = '/mnt/8TBdrive/Arstan/fibrosisprocessing/data/'

dirs = ['E10615_MYH7/', 'E10621_ABCC9/', 'E10691_RBM20/', 'E10788_LMNA/',
        'E10884/', 'E10927_MYBPC3/', 'E11442_TTN/', 'E11443_LMNA/',
        'E11444_LMNA/', 'E11971_MYH7/']


def filter(image):
    mask_min_size = morphology.remove_small_objects(image, min_size=1e2,
                                                    connectivity=1)
    mask_max_size = morphology.remove_small_objects(image, min_size=5e2,
                                                    connectivity=1)


def compute(path, file, filter=None):
    density = Density(path, file).load()
    random_map = np.random.uniform(0, 1, density.shape)
    image = random_map < density

    if filter is not None:
        image = filter(image)

    base_image = IO.load_rescaled_png(path, file)

    rgb = {'base': [226, 168, 88],
           'fibrosis': [153, 1, 2]}

    image_rgb = 255 * np.ones((*image.shape, 3), dtype='uint8')

    for i, (val_base, val_fib) in enumerate(zip(rgb['base'], rgb['fibrosis'])):
        base = 255 * np.ones_like(image, dtype='uint8')
        base[base_image > 0] = val_base
        base[image > 0] = val_fib
        image_rgb[:, :, i] = base

    IO.save_png(path, 'Generated_100_500/', file, image_rgb)


bar = tqdm(total=len(dirs))
for dirname in dirs:
    files = [f.replace('_0.npy', '') for f in os.listdir(path + dirname + 'Edges/')
             if '_0.npy' in f]
    Parallel(n_jobs=32)(delayed(compute)(path + dirname, file)
                        for file in files)
    bar.update(1)

bar.close()
