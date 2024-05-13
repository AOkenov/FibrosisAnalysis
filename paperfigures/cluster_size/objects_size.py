from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker, colors

from fibrosisanalysis.parsers.stats_loader import StatsLoader
from fibrosisanalysis.parsers.image_loader import ImageLoader


path = Path('/Users/arstanbek/Hulk/Arstan/analysis/data')
path_stats = Path(
    '/Users/arstanbek/Projects/fibrosis-workspace/fibrosisanalysis/examples/data')

# heart = 'E10691_RBM20'
# heart = 'E11444_LMNA'
# heart = 'E10927_MYBPC3'

hearts = ['E10615_MYH7', 'E10621_ABCC9', 'E10691_RBM20', 'E10788_LMNA',
          'E10884', 'E10927_MYBPC3', 'E11442_TTN', 'E11443_LMNA',
          'E11444_LMNA', 'E11971_MYH7']

objects_sizes = []
objects_densities = []
totol_pixels = []

for heart in ['E10691_RBM20', 'E11444_LMNA', 'E10927_MYBPC3']:

    # image_loader = ImageLoader(path)
    # images = image_loader.load_hearts_data([heart])

    # images = dict(zip(images['Slice'], images['Image']))

    stats_loader = StatsLoader(path_stats, subdir='Stats')
    res = stats_loader.load_hearts_data([heart])

    objects_size = res['area'].to_numpy('int')
    objects_density = 100 * res['density'].to_numpy('float')

    # myocardium = 0
    # for heart_slice in res['Slice'].unique():
    #     myocardium += np.sum(images[heart_slice] > 0)

    objects_sizes.append(objects_size)
    objects_densities.append(objects_density)
    # totol_pixels.append(myocardium)


# totol_pixels = [214361599, 81427201, 74595947]
totol_pixels = [213810399, 80645023, 74175282]


def percent_formatter(x, pos):
    return f"{x:.0f}"


def percent_formatter_one_digits(x, pos):
    return f"{x:.1f}"


def percent_formatter_two_digits(x, pos):
    return f"{x:.2f}"


fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(10, 7),
                        gridspec_kw={'width_ratios': [1, 1, 1],
                                     'height_ratios': [0.5, 0.5, 1]})

# for i in range(2):
#     for j in [1, 3, 5]:
#         axs[i, j].axis('off')

for i, label in enumerate(['A', 'B', 'C']):
    x = np.linspace(0, 100, num=25)
    y = np.geomspace(1, 10**6, num=21)

    H, yedges, xedges = np.histogram2d(objects_densities[i], objects_sizes[i],
                                       bins=(x, y), weights=objects_sizes[i])

    # H[H > 0] = np.log(H[H > 0])
    H = 100 * H / totol_pixels[i]
    H_x = H.sum(axis=0)
    H_y = H.sum(axis=1)
    H = np.ma.masked_where(H == 0, H)

    pc = axs[2, i].pcolormesh(xedges, yedges, H, cmap='inferno_r',
                              vmin=0)
    axs[2, i].set_xscale('log')
    axs[2, i].set_xticks([1e0, 1e2, 1e4, 1e6])
    axs[2, i].set_xlabel('Cluster Size')
    axs[2, i].yaxis.set_major_formatter(ticker.FuncFormatter(percent_formatter))
    axs[2, i].set_ylabel('Fibrosis, %')
    axs[2, i].set_title('C.{}'.format(i), loc='left', fontsize=14)

    if int(10 * H.max()) < 5:
        cbar_ticks = np.arange(int(10 * H.max()) + 1) / 10
    else:
        cbar_ticks = np.arange(0, int(10 * H.max()) + 1, 5) / 10

    cax = axs[2, i].inset_axes([1.05, 0.0, 0.05, 1])
    cbar = plt.colorbar(pc, cax=cax)
    cax.set_yticks(cbar_ticks)
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        percent_formatter_one_digits))

    axs[0, i].bar(yedges[:-1], width=np.diff(yedges), height=H_y, align='edge',
                  color=colors.TABLEAU_COLORS['tab:blue'])
    axs[0, i].yaxis.set_major_formatter(ticker.FuncFormatter(
        percent_formatter_one_digits))
    axs[0, i].xaxis.set_major_formatter(ticker.FuncFormatter(percent_formatter))
    axs[0, i].set_xlim(0, 100)
    axs[0, i].set_ylabel('Total Area, %')
    axs[0, i].set_xlabel('Fibrosis, %')
    axs[0, i].set_title('A.{}'.format(i), loc='left', fontsize=14)

    axs[1, i].bar(xedges[:-1], width=np.diff(xedges), height=H_x, align='edge',
                  color=colors.TABLEAU_COLORS['tab:orange'])
    axs[1, i].set_xscale('log')
    axs[1, i].set_xticks([1e0, 1e2, 1e4, 1e6])
    axs[1, i].set_xlim(1, 1e6)
    axs[1, i].set_xlabel('Cluster Size')
    axs[1, i].yaxis.set_major_formatter(ticker.FuncFormatter(
        percent_formatter_one_digits))
    axs[1, i].set_ylabel('Total Area, %')
    axs[1, i].set_title('B.{}'.format(i), loc='left', fontsize=14)

plt.tight_layout()
plt.show()


# fig = plt.figure(layout='constrained', figsize=(10, 8))

# ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
# pc = ax.pcolormesh(xedges, yedges, H, cmap='inferno_r', vmin=H[H > 0].min())
# ax.set_xscale('log')
# ax.yaxis.set_major_formatter(ticker.FuncFormatter(percent_formatter))
# ax.set_xlabel('Cluster Size')
# ax.set_ylabel('Fibrosis Percentage')
# ax.set_title('A', loc='left', fontsize=14)

# ax_cbar = ax.inset_axes([0.94, 0.05, 0.05, 0.9])
# cbar = plt.colorbar(pc, cax=ax_cbar)
# cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(
#     percent_formatter_two_digits))
# cbar.ax.yaxis.set_ticks_position('left')

# ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
# ax_histx.tick_params(axis="x", labelbottom=False)
# ax_histx.bar(xedges[:-1], width=np.diff(xedges), height=H_x, align='edge')
# ax_histx.yaxis.set_major_formatter(ticker.FuncFormatter(
#     percent_formatter_one_digits))
# ax_histx.set_ylabel('Total Area')

# ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
# ax_histy.tick_params(axis="y", labelleft=False)
# ax_histy.barh(yedges[:-1], width=H_y, height=np.diff(yedges), align='edge')
# ax_histy.xaxis.set_major_formatter(ticker.FuncFormatter(
#     percent_formatter_one_digits))
# ax_histy.set_xlabel('Total Area')

# plt.show()
