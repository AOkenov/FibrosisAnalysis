from pathlib import Path
import numpy as np
from tqdm import tqdm

from fibrosisanalysis.parsers import StatsLoader

path = Path(__file__).parent.parent.joinpath('data')
save_path = Path('/home/arstan/Downloads/dataset')

# path = save_path

hearts = ['E10615_MYH7', 'E10621_ABCC9', 'E10691_RBM20', 'E10788_LMNA',
          'E10884', 'E10927_MYBPC3', 'E11442_TTN', 'E11443_LMNA',
          'E11444_LMNA', 'E11971_MYH7']

for heart in hearts:
    slice_names = [sl.stem for sl in path.joinpath(heart, 'Stats').glob('*')]

    for slice_name in tqdm(slice_names):
        stats = StatsLoader()
        df = stats.load_slice_data(path.joinpath(heart, 'Stats', slice_name))


# import shutil


# def copy_folders(folder_list, destination):
#     for folder in folder_list:
#         try:
#             shutil.copytree(folder, destination)
#             print(f"Folder '{folder}' copied successfully.")
#         except Exception as e:
#             print(f"Error copying folder '{folder}': {e}")



# for heart in hearts:
#     source_folder = path.joinpath(heart, 'Stats')
#     destination_folder = save_path.joinpath(heart, 'Stats')
#     copy_folders([source_folder], destination_folder)
