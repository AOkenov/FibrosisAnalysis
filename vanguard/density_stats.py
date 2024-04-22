import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from tqdm.notebook import tqdm
from skimage import morphology, measure, filters
from matplotlib import rcParams
from joblib import Parallel, delayed
import matplotlib as mpl
config = {
    'figure.figsize': [3.3, 2.7],
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 7,
    'axes.linewidth': 0.3,
    'lines.linewidth': 1,
    'lines.markersize': 1,
    'grid.linewidth': 0.5,
    'xtick.labelsize': 5,
    'xtick.major.pad': 1,
    'xtick.major.size': 2,
    'ytick.labelsize': 5,
    'ytick.major.pad': 1,
    'ytick.major.size': 2}
rcParams.update(config)


step = 0.01
dens_bins = np.arange(step/2, 1, step)
max_areas = np.zeros((len(dens_bins), 100))

df = []


def compute(dens):
    size = 400
    binary = np.random.rand(size, size)
    binary = binary <= dens
#    binary = morphology.remove_small_objects(binary, min_size=10)
    labeled = morphology.label(binary, connectivity=1)
    props = measure.regionprops_table(labeled, properties=['label', 'area'])
    props = pd.DataFrame(props)
    props['density'] = dens
    return props


bar = tqdm(total=dens_bins.shape[0])
for i, dens in enumerate(dens_bins):
    props = Parallel(n_jobs=8)(delayed(compute)(dens) for j in range(100))
    df += props
    bar.update(1)
bar.close()

df = pd.concat(df)

df.to_pickle('/home/venus/Projects/heartrecon/wetransfer-21bce0/dens_gen.csv')
