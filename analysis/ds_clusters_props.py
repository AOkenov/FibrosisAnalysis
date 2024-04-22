from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


from fibrosisanalysis.parsers import ImageLoader
from fibrosisanalysis.analysis import ObjectsPropertiesBuilder


path = Path(__file__).parent.joinpath('data')

original_images = [f.stem for f in path.joinpath('original_texs').glob('*.png')]

for f in original_images[:1]:
    image_loader = ImageLoader()
    original = image_loader.load_slice_data(path.joinpath('original_texs', f))
    replicated = np.load(path.joinpath('sim_dir_2', f).with_suffix('.npy'))

    original_obj = ObjectsPropertiesBuilder().build_from_segment(original > 1)
    replicated_obj = ObjectsPropertiesBuilder().build_from_segment(replicated > 1)

    print(original_obj.orientation)
    print(replicated_obj.orientation)

    fig, axs = plt.subplots(ncols=2)
    axs[0].imshow(original, cmap='viridis')
    axs[1].imshow(replicated, cmap='viridis')
    plt.show()

    fig, axs = plt.subplots(ncols=2)
    axs[0].hist(original_obj.axis_ratio, bins=20)
    axs[1].hist(replicated_obj.axis_ratio, bins=20)
    plt.show()
