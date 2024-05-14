from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib import patches
from skimage import morphology

from fibrosisanalysis.slice.heart_slice import HeartSliceBuilder
from fibrosisanalysis.parsers import ImageLoader, DensityLoader


cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'fibrosis', [(0, 'white'),
                 (0.5, '#e2a858'),
                 (1, '#990102')])

# path = Path(__file__).parents[1].joinpath('data')
path = Path('/Users/arstanbek/Library/CloudStorage/OneDrive-UGent/data')

heart = 'E11444_LMNA'
slice_name = 'E11444_08_SC2'

image_loader = ImageLoader(path)
image = image_loader.load_slice_data(path.joinpath(heart, 'Images',
                                                   slice_name))
image_gen = image_loader.load_slice_data(path.joinpath(heart, 'Generated',
                                                       slice_name))

density_loader = DensityLoader(path)
density_map = density_loader.load_slice_data(path.joinpath(heart, 'Density',
                                                           slice_name))


def draw_lines(fig, ax0, ax1, x0, y0, x1, y1):
    fig.canvas.draw()
    transFigure = fig.transFigure.inverted()

    x0, y0 = transFigure.transform(ax0.transData.transform([x0, y0]))
    x1, y1 = transFigure.transform(ax1.transData.transform([x1, y1]))

    line = mpl.lines.Line2D((x0, x1), (y0, y1),
                            transform=fig.transFigure, lw=0.5, color='black')

    fig.lines.append(line)


def show_zoom(image, ax_base, ax_zoom, x, y, dx, dy, cmap='gray', title='',
              vmin=0, vmax=2):
    image_0 = image[y: y + dy, x: x + dx]
    ax_zoom.imshow(image_0, cmap=cmap, origin='lower', aspect='equal',
                   vmin=vmin, vmax=vmax)
    ax_zoom.set_title(title, loc='left', fontsize=12)
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])

    rect = patches.Rectangle((x, y), dx, dy, linewidth=1,
                             edgecolor='black', facecolor='none')
    ax_base.add_patch(rect)


def percentile_formatter(x, pos):
    return f'{int(x * 100)}%'


y0 = 2300
dy0 = 300
x0 = 870
dx0 = 300

y0 = 2400
dy0 = 100
x0 = 970
dx0 = 100

fig = plt.figure(figsize=(8, 4))
axs = fig.subplot_mosaic([
    ["szoom", "szoom", "empty", "dzoom", "dzoom", "dbar", "gzoom", "gzoom", "empty2"],
    ["szoom", "szoom", "empty", "dzoom", "dzoom", "dbar", "gzoom", "gzoom", "empty2"],
    ["slice", "slice", "slice", "density", "density", "density", "generated", "generated", "generated"],
    ["slice", "slice", "slice", "density", "density", "density", "generated", "generated", "generated"],
    ["slice", "slice", "slice", "density", "density", "density", "generated", "generated", "generated"],
    # ["empty3", "empty3", "empty3", "dbar", "dbar", "dbar", "empty4", "empty4", "empty4"],
], height_ratios=[1, 1, 1, 1, 1])

for k, ax in axs.items():
    ax.set_anchor('C')

for k, v in axs.items():
    if 'empty' in k:
        v.axis('off')

axs["slice"].imshow(image, cmap=cmap, origin='lower', aspect='equal',
                    vmin=0, vmax=2)
axs["slice"].axis('off')

show_zoom(image, axs["slice"], axs["szoom"], x0, y0, dx0, dy0, cmap,
          title='A. Histological')

axs["generated"].imshow(image_gen, cmap=cmap, origin='lower', aspect='equal',
                        vmin=0, vmax=2)
axs["generated"].axis('off')

show_zoom(image_gen, axs["generated"], axs["gzoom"], x0, y0, dx0, dy0, cmap,
          title='B. Uniform Generator')

print(density_map.max())

mask = morphology.remove_small_objects(image == 0, min_size=10_000)
density_map = np.ma.masked_where(mask, density_map)
# density_map = np.ma.masked_where(image == 0, density_map)
dmap = axs["density"].imshow(density_map, cmap='inferno_r', origin='lower',
                             aspect='equal', vmin=0, vmax=0.8)
axs["density"].axis('off')

show_zoom(density_map, axs["density"], axs["dzoom"], x0, y0, dx0, dy0,
          cmap='inferno_r', title='C. Density Map', vmin=0, vmax=0.8)

axs['dbar'].axis('off')
cbar = fig.colorbar(dmap, orientation='vertical', ax=axs['dbar'],
                    fraction=0.5, pad=0.05, use_gridspec=True)
cbar.set_label('Fibrosis, %')
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8])
cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(percentile_formatter))
cbar.ax.yaxis.set_ticks_position('left')

plt.tight_layout()

for base, zoom in zip(['slice', 'density', 'generated'],
                      ['szoom', 'dzoom', 'gzoom']):
    draw_lines(fig, axs[base], axs[zoom], x0, y0, 0, 0)
    draw_lines(fig, axs[base], axs[zoom], x0 + dx0, y0, dx0, 0)

plt.show()

fig.savefig('paperfigures/figures/uniform_generator.png', dpi=300)
