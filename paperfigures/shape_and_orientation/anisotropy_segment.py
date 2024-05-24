from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, colormaps
from skimage import morphology, segmentation

from bitis.texture.properties import (
    DistributionEllipseBuilder, 
    PolarPlots,
    PatternPropertiesBuilder
)
from fibrosisanalysis.parsers import ImageLoader


cmap = colors.LinearSegmentedColormap.from_list(
    'fibrosis', [(0, 'white'),
                 (0.5, '#e2a858'),
                 (1, '#990102')])

cmap_2 = colors.LinearSegmentedColormap.from_list(
    'fibrosis', [(0, '#e2a858'),
                 (1, '#990102')])


def swap_axis(angle):
    """
    Swap axis for polar plot.
    """
    return 0.5 * np.pi - angle


def draw_anisotropy(ax, objects_props, n_std=2):
    dist_ellipse_builder = DistributionEllipseBuilder()
    dist_ellipse_builder.build(objects_props)
    dist_ellipse = dist_ellipse_builder.distribution_ellipse
    full_theta = swap_axis(dist_ellipse.full_theta)
    orientation = swap_axis(dist_ellipse.orientation)
    r, theta, d = PolarPlots.sort_by_density(objects_props.axis_ratio,
                                             swap_axis(objects_props.orientation))

    ax.scatter(theta, r, c=d, s=30, alpha=1, cmap='viridis')
    ax.plot(full_theta, dist_ellipse.full_radius, color='red')

    ax.quiver(0, 0, orientation, 0.5 * dist_ellipse.width,
              angles='xy', scale_units='xy', scale=1, color='red')
    ax.quiver(0, 0, 0.5 * np.pi + orientation,
              0.5 * dist_ellipse.height,
              angles='xy', scale_units='xy', scale=1, color='red')


def draw_segment(ax, image, objects_props):

    print(objects_props.head())

    ax.imshow(image, cmap=cmap, origin='lower')

    _, _, density = PolarPlots.point_density(objects_props['axis_ratio'],
                                             objects_props['orientation'])

    density_cmap = colormaps.get_cmap('viridis')

    for i in range(len(objects_props['major_axis_length'])):
        color = density_cmap(density[i] / density.max())
        width = objects_props['major_axis_length'][i]
        height = objects_props['minor_axis_length'][i]
        alpha = objects_props['orientation'][i]
        centroids = objects_props[['centroid-0', 'centroid-1']].to_numpy(int)
        xy = centroids[i]

        res = PolarPlots.rotated_ellipse(width, height, 0.5 * np.pi - alpha)
        y, x = PolarPlots.polar_to_cartesian(*res)
        y += xy[1]
        x += xy[0]
        ax.plot(y, x, color=color, lw=1)

    ax.axis('off')


path = Path(__file__).parent.parent.parent.joinpath('data')
path_stats = Path(__file__).parent.parent.parent.joinpath('data')

heart = 'E11444_LMNA'
slice_name = 'E11444_08_SC2'

image_loader = ImageLoader()
image = image_loader.load_slice_data(path.joinpath(heart, 'Images',
                                                   slice_name))

image = image[2400:2500, 970:1070]
mask = image == 2

objects_props_builder = PatternPropertiesBuilder(area_min=10)
pattern_props = objects_props_builder.build(mask, clear_border=True)
objects_props = pattern_props.objects_props

mask = morphology.remove_small_objects(mask, 10)
mask = segmentation.clear_border(mask)
image[(mask == 0) & (image > 0)] = 1

n_std = 2

fig = plt.figure(figsize=(11, 5))
gs = fig.add_gridspec(1, 2, left=0.05, right=0.95,
                      bottom=0.05, top=0.9, wspace=0.05, hspace=0.2)
axs = []
axs.append(fig.add_subplot(gs[0]))
draw_segment(axs[0], image, objects_props)

axs.append(fig.add_subplot(gs[1], projection='polar'))
draw_anisotropy(axs[-1], objects_props, n_std)

axs[0].set_title('A', loc='left', fontsize=14, y=1)
axs[1].set_title('B', loc='left', fontsize=14, y=0.985)
plt.show()

# fig.savefig('paperfigures/figures/segment_anisotropy.png', dpi=300,
#             bbox_inches='tight')
