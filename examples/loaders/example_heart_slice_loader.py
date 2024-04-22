from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from tqdm import tqdm

from fibrosisanalysis.parsers.image_loader import ImageLoader


path = Path(__file__).parent.parent.parent.joinpath('data')


image_loader = ImageLoader()
hearts = ['E10615_MYH7', 'E10621_ABCC9', 'E10691_RBM20', 'E10788_LMNA',
          'E10884', 'E10927_MYBPC3', 'E11442_TTN', 'E11443_LMNA',
          'E11444_LMNA', 'E11971_MYH7']

for heart in hearts[:]:
    edge_path = path.joinpath(heart, 'Edges')
    file_names = [f.name.replace('_0.npy', '') for f in edge_path.iterdir()
                  if f.is_file() and '_0.npy' in f.name]
    print(heart)
    for file_name in file_names:
        # file_name = file_name.replace('_WABL', '_NABL')
        image = image_loader.load_slice_data(path.joinpath(heart, 'Images',
                                                           file_name))
        density = image_loader.load_slice_data(path.joinpath(heart, 'Density',
                                                             file_name))
        if image.shape != density.shape:
            print(f'{heart}/{file_name} has different shapes')
            print(f'Image shape: {image.shape}, density shape: {density.shape}')