import sys
sys.path.append('/mnt/8TBdrive/Arstan/')
sys.path.append('/mnt/8TBdrive/Arstan/Finitewave/')

from tqdm import tqdm
from skimage import morphology	
from joblib import Parallel, delayed
import os

from fibrosisprocessing.slice.slice import Slice
from fibrosisprocessing.slice.objects_properties import ObjectsProperties
from fibrosisprocessing.tools.io import IO
path = '/mnt/8TBdrive/Arstan/analysis/data/'

dirs = ['E10615_MYH7/', 'E10621_ABCC9/', 'E10691_RBM20/', 'E10788_LMNA/',
        'E10884/', 'E10927_MYBPC3/', 'E11442_TTN/', 'E11443_LMNA/',
        'E11444_LMNA/', 'E11971_MYH7/']


def filter_objects(labels):
    mask = morphology.remove_small_objects(labels > 0, min_size=1e2,
                                           connectivity=1)

    labels[mask == 0] = 0
    return labels


def compute(path, file):
    # load data for slice
    slc = Slice(path, file, dirname='Generated/')
    # compute properties of individual objects on slice
    props_list = ['label', 'area', 'centroid', 'major_axis_length',
                  'minor_axis_length', 'orientation', 'solidity']
    labels = slc.label(slc.image == 2, slc.mask)
    obj_props = ObjectsProperties.compute(labels, slc.density_map, props_list,
                                          extra_props=[ObjectsProperties.density])

    coords = obj_props[['centroid-0', 'centroid-1']].to_numpy()
    # compute properties of epicardial edges
    edge_props_list = ['label', 'centroid', 'orientation', 'major_axis_length']
    epi_props = EdgeProperties.compute(slc.mask, slc.edges['epi'],
                                       edge_props_list)
    obj_props['distance'] = ObjectsProperties.distance(coords, slc.edges)
    obj_props['tangent'] = ObjectsProperties.tangent(coords, epi_props)

    IO.save_df(slc.path, 'StatsGenerated/', file, obj_props)


def compute_additional_properties(path, file):
    # load data for slice
    slc = Slice(path, file, dirname='Generated/')
    # compute properties of individual objects on slice
    props_list = ['label', 'image']
    labels = slc.label(slc.image == 2, slc.mask)
    labels = filter_objects(labels)
    obj_props = ObjectsProperties.compute(labels, slc.density_map, props_list,
                                          extra_props=[ObjectsProperties.density])

    obj_props_old = IO.load_df(path, 'StatsGenerated/', file)

    obj_props = slc.merge_props(obj_props, obj_props_old)

    IO.save_df(slc.path, 'StatsGenerated/', file, obj_props)


bar = tqdm(total=len(dirs))
for dirname in dirs[:]:
    files = [f.replace('_0.npy', '') for f in os.listdir(path + dirname + 'Edges/')
             if '_0.npy' in f]
    Parallel(n_jobs=16)(delayed(compute_additional_properties)(path + dirname, file)
                        for file in files)
    bar.update(1)

bar.close()
