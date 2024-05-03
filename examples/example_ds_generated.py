from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


from fibrosisanalysis.parsers import ImageLoader
from fibrosisanalysis.analysis import ObjectsPropertiesBuilder

cmap = colors.LinearSegmentedColormap.from_list(
    'fibrosis', [(0, 'white'),
                 (0.5, '#e2a858'),
                 (1, '#990102')])


path = Path(__file__).parent.parent.joinpath('analysis', 'data')

print(path)

original_images = [f.stem for f in path.joinpath('original_texs').glob('*.png')]


for f in original_images[:5]:
    image_loader = ImageLoader()
    original = image_loader.load_slice_data(path.joinpath('original_texs', f))
    replicated = np.load(path.joinpath('sim_dir_2', f).with_suffix('.npy'))

    original_obj = ObjectsPropertiesBuilder().build_from_segment(original > 1)
    replicated_obj = ObjectsPropertiesBuilder().build_from_segment(replicated > 1)

    # print(original_obj.orientation)
    # print(replicated_obj.orientation)

    fig, axs = plt.subplots(ncols=2)
    axs[0].imshow(original, cmap=cmap, origin='lower', vmin=0, vmax=2)
    axs[0].set_title('Original')
    axs[1].imshow(replicated, cmap=cmap, origin='lower', vmin=0, vmax=2)
    axs[1].set_title('Replicated')
    plt.show()

    fig.savefig(f'{f}.png', dpi=300)

    # fig, axs = plt.subplots(ncols=2)
    # axs[0].hist(original_obj.axis_ratio, bins=20)
    # axs[1].hist(replicated_obj.axis_ratio, bins=20)
    # plt.show()
