from pathlib import Path
import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from matplotlib import ticker, patches


from fibrosisanalysis.parsers import ImageLoader, DensityLoader


path = Path(__file__).parent.parent.joinpath('data')

hearts = [f.name for f in path.glob('*') if f.is_dir()]
example_hearts = {'E10691_RBM20': 'A',
                  'E11444_LMNA': 'B',
                  'E10927_MYBPC3': 'C'}

density_per_heart = []
mean_density_per_heart = {}

for heart in hearts[:]:
    slice_names = Path(path.joinpath(heart, 'Stats')).glob('*')
    slice_names = sorted([f.stem for f in slice_names if f.suffix == '.pkl'])

    density = []

    total_pixels = 0
    fibrotic_pixels = 0

    for slice_name in tqdm(slice_names[:]):
        slice_name = f'{slice_name}.png'
        image_loader = ImageLoader()
        image = image_loader.load_slice_data(path.joinpath(heart, 'Images',
                                                           slice_name))
        image = image.astype(np.uint8)

        density.append(np.sum(image == 2) / np.sum(image > 0))
        fibrotic_pixels += np.sum(image == 2)
        total_pixels += np.sum(image > 0)

        gc.collect()

    df = pd.DataFrame({'Density': np.array(density),
                       'Slice Index': np.arange(len(density)),
                       'Heart': np.repeat(heart, len(density))})

    density_per_heart.append(df)

    mean_density_per_heart[heart] = fibrotic_pixels / total_pixels


sorted_hearts = sorted(mean_density_per_heart.keys(),
                       key=lambda x: mean_density_per_heart[x])

df = pd.concat(density_per_heart)


def percent_formatter(x, pos):
    return f'{int(100 * x)}%'


fig = plt.figure(figsize=(8, 8.5))
axs = fig.subplot_mosaic([
    [sorted_hearts[0], 'empty', 'empty'],
    sorted_hearts[1:4],
    sorted_hearts[4:7],
    sorted_hearts[7:],
])

axs['empty'].set_visible(False)

for i, heart in enumerate(sorted_hearts):
    ddf = df[df['Heart'] == heart]

    slice_index = ddf['Slice Index'] + 1

    fibrotic = ddf['Density']
    non_fibrotic = 1 - fibrotic

    ax = axs[heart]

    ax.bar(slice_index, fibrotic, color='red', alpha=0.5, width=1, ec='black')
    # ax.bar(slice_index, non_fibrotic, bottom=fibrotic,
    #                color='blue', alpha=0.5, width=1, ec='black')
    ax.set_xticks([1, 5, 10, 15])
    ax.sharex(axs[sorted_hearts[0]])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(percent_formatter))

    if heart in sorted_hearts[7:]:
        ax.set_xlabel('Slice Index')

    if heart in [*sorted_hearts[:2], sorted_hearts[4], sorted_hearts[7]]:
        ax.set_ylabel('Fibrotic Pixels')

    if heart in example_hearts:
        for loc in ['top', 'right', 'left', 'bottom']:
            ax.spines[loc].set_color('black')
            ax.spines[loc].set_linewidth(2)
        ax.add_patch(patches.Rectangle((0., 0.85), 0.13, 0.15, fill=False,
                                       edgecolor='black', lw=2,
                                       transform=ax.transAxes))
        ax.text(0.06, 0.91, example_hearts[heart], transform=ax.transAxes,
                ha='center', va='center', fontsize=12)

    ax.set_ylim(0, 1.2 * np.max(fibrotic))

    heart_index = i + 1
    ax.text(0.94, 0.91, f'{heart_index}', transform=ax.transAxes,
            ha='center', va='center', fontsize=12)
    ax.add_patch(patches.Rectangle((0.87, 0.85), 0.13, 0.15, fill=False,
                                   edgecolor='black', lw=1,
                                   transform=ax.transAxes))

plt.subplots_adjust(top=0.95, bottom=0.1, right=0.98, left=0.1,
                    wspace=0.3, hspace=0.2)
fig.savefig('paperfigures/figures/density_per_slice.png', dpi=300)
plt.show()
