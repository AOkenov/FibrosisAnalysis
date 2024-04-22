from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, patches
from scipy import spatial
from skimage import measure
from fibrosisanalysis.parsers.stats_loader import StatsLoader


def convex_hull(image):
    coords = np.argwhere(image)
    hull = spatial.ConvexHull(coords)
    return coords[list(hull.vertices) + [hull.vertices[0]]]


def equvivalent_ellipse(image):
    props = measure.regionprops(image.astype(int))
    major_axis = props[0].axis_major_length
    minor_axis = props[0].axis_minor_length
    angle = props[0].orientation
    return major_axis, minor_axis, 0.5 * np.pi - angle


cmap = colors.LinearSegmentedColormap.from_list(
    'fibrosis', [(0, 'white'),
                 (0.5, '#e2a858'),
                 (1, '#990102')])


path = Path(
    '/Users/arstanbek/Projects/fibrosis-workspace/fibrosisanalysis/examples/data')

heart = 'E11444_LMNA'
slice_name = 'E11444_08_SC2'

stats_loader = StatsLoader(path)
path = path.joinpath(heart, 'Stats', slice_name)
res = stats_loader.load_slice_data(path)

res = res[(res['area'].between(50, 100))]

# print(res.info())

fig, axs = plt.subplots(ncols=5, nrows=3, sharey=True,
                        width_ratios=[0.2, 1, 1, 1, 1],
                        figsize=(11, 8))
for i, n in enumerate([8, 20, 3, 36]):
    image = np.ones((20, 20))
    image_data = res.iloc[n]['image']

    dx = image.shape[0] - image_data.shape[0]
    dy = image.shape[1] - image_data.shape[1]

    image[dx // 2: dx // 2 + image_data.shape[0],
          dy // 2: dy // 2 + image_data.shape[1]] = 1 + image_data

    # convex_hull = morphology.convex_hull_image(image > 1)

    for j in range(3):
        ax = axs[j, i + 1]
        ax.imshow(image, cmap=cmap, vmin=0, vmax=2)
        ax.axis('off')
        axs[j, 0].axis('off')

    for j, label in enumerate(['A', 'B', 'C']):
        axs[j, 0].axis('off')
        axs[j, 0].text(0.5, 1., label, rotation='horizontal', fontsize=16)

    hull_vertices = convex_hull(image > 1)

    axs[1, i + 1].plot(hull_vertices[:, 1],
                       hull_vertices[:, 0], '#1f77b4', lw=3)

    width, height, angle = equvivalent_ellipse(image > 1)
    xy = np.argwhere(image > 1).mean(axis=0)
    patch = patches.Ellipse(xy=xy[::-1], width=width, height=height,
                            angle=np.degrees(angle), facecolor='none',
                            edgecolor='#2ca02c', lw=3)

    axs[2, i + 1].add_patch(patch)

    U = 0.5 * width * np.sin(1.5 * np.pi - angle)
    V = - 0.5 * width * np.cos(1.5 * np.pi - angle)

    axs[2, i + 1].quiver(xy[1], xy[0], U, V, scale_units='xy', scale=1,
                         color='black', width=0.015)

    U = 0.5 * height * np.sin(- angle)
    V = - 0.5 * height * np.cos(- angle)

    axs[2, i + 1].quiver(xy[1], xy[0], U, V, scale_units='xy', scale=1,
                         color='black', width=0.015)

plt.tight_layout()
plt.show()

fig.savefig('paperfigures/figures/fibrosis_structures.png', dpi=300)
