import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

from fibrosisprocessing.tools import IO, Edge
from fibrosisprocessing.segmentation import Segmentation


N_RADIAL = 6
N_ANGULAR = 72

PATH = '/home/venus/Projects/heartrecon/PNG/'

PATH_SAVE = PATH + 'Segments/{}x{}/'.format(N_RADIAL, N_ANGULAR)

if not os.path.isdir(PATH_SAVE):
    os.makedirs(PATH_SAVE)


files = sorted([f.split('_0.npy')[0] for f in os.listdir(PATH + 'Edges/')
                if '_0.npy' in f])[12:13]


def compute(path, file):
    edges = {}
    edges['epi'] = Edge(IO.load_edge(path, file, '_0'))
    edges['endo'] = Edge(IO.load_edge(path, file, '_1'))

    image = IO.load_png(path, file, ablation='WABL')

    segmentation = Segmentation(edges, image)
    labeled = segmentation.label(N_RADIAL, N_ANGULAR)
    np.save(PATH_SAVE + file + '.npy', labeled.astype(np.uint16))
    # return image, labeled


bar = tqdm(total=len(files))
for file in files:
    compute(PATH, file)
    bar.update()
bar.close()

# file = files[5]
# image, labeled = compute(PATH, file)
# fig, axs = plt.subplots(ncols=2)
# axs[0].imshow(image, origin='lower')
# axs[1].imshow(labeled, origin='lower')
# plt.show()
