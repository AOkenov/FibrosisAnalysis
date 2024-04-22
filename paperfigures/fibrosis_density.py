from pathlib import Path
import gc
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from matplotlib import ticker


from fibrosisanalysis.parsers import ImageLoader, DensityLoader


path = Path(__file__).parent.parent.joinpath('data')

hearts = [f.name for f in path.glob('*') if f.is_dir()]

# Density in non-fibrotic and fibrotic pixels
nonfibrotic_density_per_heart = {}
fibrotic_density_per_heart = {}
# Mean density of fibrotic pixels in the heart
mean_density_per_heart = {}
number_of_slices_per_heart = {}

for heart in hearts[:]:
    slice_names = Path(path.joinpath(heart, 'Stats')).glob('*')
    slice_names = sorted([f.stem for f in slice_names if f.suffix == '.pkl'])

    density_fibrotic = np.zeros(20, dtype='float')
    density_nonfibrotic = np.zeros(20, dtype='float')
    density_bins = np.linspace(0, 1, 21)

    total_pixels = 0
    fibrotic_pixels = 0

    for slice_name in tqdm(slice_names[:]):
        slice_name = f'{slice_name}.png'
        density_loader = DensityLoader()
        density_map = density_loader.load_slice_data(path.joinpath(heart,
                                                                   'Density',
                                                                   slice_name))
        image_loader = ImageLoader()
        image = image_loader.load_slice_data(path.joinpath(heart, 'Images',
                                                           slice_name))
        image = image.astype(np.uint8)

        if density_map.shape != image.shape:
            print(f'Slice {slice_name} has different shapes')
            continue

        fibrotic = np.histogram(density_map[image == 2], bins=density_bins)[0]
        nonfibrotic = np.histogram(density_map[image == 1],
                                   bins=density_bins)[0]
        density_fibrotic += fibrotic
        density_nonfibrotic += nonfibrotic

        fibrotic_pixels += np.sum(image == 2)
        total_pixels += np.sum(image > 0)
        gc.collect()

    fibrotic_density_per_heart[heart] = density_fibrotic / total_pixels
    nonfibrotic_density_per_heart[heart] = density_nonfibrotic / total_pixels
    mean_density_per_heart[heart] = fibrotic_pixels / total_pixels
    number_of_slices_per_heart[heart] = len(slice_names)


sorted_hearts = sorted(mean_density_per_heart.keys(),
                       key=lambda x: mean_density_per_heart[x])


def percent_formatter(x, pos):
    return f"{int(100 * x)}%"


fig, axs = plt.subplots(ncols=5, nrows=2, sharex=True,
                        figsize=(10, 4))

for i, heart in enumerate(sorted_hearts):
    print(f'{heart}' + f' {number_of_slices_per_heart[heart]}'
          + f' {mean_density_per_heart[heart]}')
    ax = axs[i // 5, i % 5]
    fibrotic_bar = ax.bar(density_bins[:-1], fibrotic_density_per_heart[heart],
                          width=0.05, align='edge')
    nonfibrotic_bar = ax.bar(density_bins[:-1],
                             nonfibrotic_density_per_heart[heart],
                             bottom=fibrotic_density_per_heart[heart],
                             width=0.05, align='edge')
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(percent_formatter))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(percent_formatter))

    if i % 5 == 0:
        ax.set_ylabel('Fraction of Pixels, %')

    if i >= 5:
        ax.set_xlabel('Fibrosis, %')

plt.subplots_adjust(top=0.8, bottom=0.15, right=0.98, left=0.1,
                    wspace=0.3, hspace=0.2)

fig.legend(handles=nonfibrotic_bar.patches[:1] + fibrotic_bar.patches[:1],
           labels=['Non-Fibrotic', 'Fibrotic'], loc='center',
           title='Pixel Type',
           bbox_to_anchor=(0.5, 0.9), ncol=2)
plt.show()

path_save = Path(__file__).parent.joinpath('figures')
fig.savefig(path_save.joinpath('fibrosis_density.png'),
            dpi=300, bbox_inches='tight')
