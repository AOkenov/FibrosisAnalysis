from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker, colors
from tqdm import tqdm

from fibrosisanalysis.parsers import ImageLoader, StatsLoader


path = Path(__file__).parent.parent.joinpath('data')


hearts = {'E10691_RBM20': 213810399, 'E11444_LMNA': 80645023,
          'E10927_MYBPC3': 74175282}

density_bins = [-0.01, 0.2, 0.4, 1.01]
size_bins = np.geomspace(1, 10**6, num=13)


def make_hist(x, labels, bins, weight=1):
    out = []
    for i in range(1, len(density_bins)):
        mask = labels == i
        if np.all(mask == False):
            out.append(np.zeros(len(bins) - 1))
            continue

        out.append(np.histogram(x[mask], weights=x[mask] / weight, bins=bins)[0])

    return out


fraction_of_pixels = {}
fraction_of_pixels_gen = {}
# total_pixels = []

for heart, total_pixels in hearts.items():

    slice_names = Path(path.joinpath(heart, 'Stats')).glob('*')
    slice_names = sorted([f.stem for f in slice_names if f.suffix == '.pkl'])

    # total_pixels = 0
    objects_size = []
    objects_densities = []

    objects_size_gen = []
    objects_densities_gen = []

    for slice_name in tqdm(slice_names[:]):
        # image_loader = ImageLoader()
        # image = image_loader.load_slice_data(path.joinpath(heart, 'Images',
        #                                                    slice_name))
        # image = image.astype(int)
        # total_pixels += np.sum(image > 0)

        stats_loader = StatsLoader('')
        stats = stats_loader.load_slice_data(path.joinpath(heart, 'Stats',
                                                           slice_name))

        objects_size.append(stats['area'].to_numpy('int'))
        objects_densities.append(stats['density'].to_numpy('float'))

        stats = stats_loader.load_slice_data(path.joinpath(heart,
                                                           'StatsGenerated',
                                                           slice_name))
        objects_size_gen.append(stats['area'].to_numpy('int'))
        objects_densities_gen.append(stats['density'].to_numpy('float'))

    # print(f'{heart}: {total_pixels}')

    objects_size = np.concatenate(objects_size)
    objects_densities = np.concatenate(objects_densities)

    labels = np.digitize(objects_densities, bins=density_bins)
    fraction_of_pixels[heart] = make_hist(objects_size, labels,
                                          bins=size_bins,
                                          weight=hearts[heart])

    objects_size_gen = np.concatenate(objects_size_gen)
    objects_densities_gen = np.concatenate(objects_densities_gen)

    labels = np.digitize(objects_densities_gen, bins=density_bins)
    fraction_of_pixels_gen[heart] = make_hist(objects_size_gen, labels,
                                              bins=size_bins,
                                              weight=hearts[heart])

    # objects_sizes[heart] = np.concatenate(objects_size)
    # objects_sizes_gen[heart] = np.concatenate(objects_size_gen)


def percent_formatter(x, pos):
    x = 100 * x
    return f"{x:.1f}%"


def plot_bar(ax, height, bins, color='blue'):
    bar = ax.bar(bins[:-1], width=np.diff(bins), height=height, align='edge',
                 alpha=0.5, edgecolor='black', color=color)
    ax.set_xscale('log')
    ax.set_xticks([1e0, 1e2, 1e4, 1e6])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(percent_formatter))
    return bar


fig, axs = plt.subplots(ncols=4, nrows=len(density_bins),
                        sharex=True,
                        gridspec_kw={'width_ratios': [1, 1, 1, 0.15]},
                        figsize=(8, 7))

for j in range(4):
    axs[j, -1].set_axis_off()


axs[0, -1].text(0., 0.5, '<20%', transform=axs[0, -1].transAxes, ha='center',
                va='center', rotation=0)
axs[1, -1].text(0., 0.5, '20 - 40%', transform=axs[1, -1].transAxes,
                ha='center', va='center', rotation=0)
axs[2, -1].text(0., 0.5, r'$\geq$40%', transform=axs[2, -1].transAxes,
                ha='center', va='center', rotation=0)
axs[3, -1].text(0., 0.5, '0 - 100%', transform=axs[3, -1].transAxes,
                ha='center', va='center', rotation=0)

axs[0, 0].set_title('A', loc='left')
axs[0, 1].set_title('B', loc='left')
axs[0, 2].set_title('C', loc='left')
axs[0, 3].set_title('Fibrosis', loc='right')


for i, (heart, _) in enumerate(hearts.items()):
    pixels = fraction_of_pixels_gen[heart]
    for j, size in enumerate(pixels):
        plot_bar(axs[j, i], size, size_bins, color='red')
        if i == 0:
            axs[j, i].set_ylabel('Fraction of Pixels')

    bar_gen = plot_bar(axs[j+1, i], np.sum(pixels, axis=0), size_bins,
                       color='red')
    axs[j+1, i].set_xlabel('Cluster Size')

    pixels = fraction_of_pixels[heart]
    for j, size in enumerate(pixels):
        plot_bar(axs[j, i], size, size_bins, color='blue')

    bar = plot_bar(axs[j+1, i], np.sum(pixels, axis=0), size_bins,
                   color='blue')
    axs[j+1, i].set_xlabel('Cluster Size')

    if i == 0:
        axs[j+1, i].set_ylabel('Fraction of Pixels')


plt.subplots_adjust(top=0.9, bottom=0.1, right=0.98, left=0.1,
                    wspace=0.35, hspace=0.1)

fig.legend(handles=bar_gen.patches[:1] + bar.patches[:1],
           labels=['Uniform generator', 'Histology'], loc='center',
           bbox_to_anchor=(0.5, 0.96), ncol=2)
plt.show()

path_save = Path(__file__).parent.joinpath('figures')
fig.savefig(path_save.joinpath('cluster_size_distribution.png'),
            dpi=300, bbox_inches='tight')
