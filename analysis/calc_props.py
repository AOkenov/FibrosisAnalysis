from pathlib import Path
import pandas as pd
from tqdm import tqdm
from skimage import measure

from fibrosisanalysis.parsers import ImageLoader


def calc_props(image, props_list):
    mask = (image == 2).astype(int)
    labels = measure.label(mask, background=0, connectivity=1)
    props = measure.regionprops_table(labels, properties=props_list)
    return pd.DataFrame(props)


props_list = ['label', 'area', 'centroid', 'solidity', 'major_axis_length',
              'minor_axis_length', 'orientation', 'perimeter']


path = Path('/Users/arstanbek/Library/CloudStorage/OneDrive-UGent/data/')
hearts = ['E10691_RBM20', 'E11444_LMNA', 'E10927_MYBPC3']

for heart in hearts:
    image_loader = ImageLoader(path)

    slice_names = list(path.joinpath(heart, 'DS').glob('*.png'))
    slice_names = [slice_name.stem for slice_name in slice_names]

    if not path.joinpath(heart, 'StatsDS').exists():
        path.joinpath(heart, 'StatsDS').mkdir()

    for slice_name in tqdm(slice_names):
        image = image_loader.load_slice_data(path.joinpath(heart, 'DS',
                                                           slice_name))
        props = calc_props(image, props_list)

        pd.to_pickle(props, path.joinpath(heart, 'StatsDS',
                                          slice_name).with_suffix('.pkl'))
