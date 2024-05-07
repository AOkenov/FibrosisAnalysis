from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, colormaps
from skimage import morphology, segmentation

from fibrosisanalysis.slice.heart_slice import HeartSliceBuilder
from fibrosisanalysis.analysis.objects_properties import (
    ObjectsPropertiesBuilder
)
from fibrosisanalysis.analysis.distribution_ellipses import (
    DistributionEllipseBuilder
)
from fibrosisanalysis.plots.polar_plots import PolarPlots
from fibrosisanalysis.parsers import StatsLoader


cmap = colors.LinearSegmentedColormap.from_list(
    'fibrosis', [(0, 'white'),
                 (0.5, '#e2a858'),
                 (1, '#990102')])

cmap_2 = colors.LinearSegmentedColormap.from_list(
    'fibrosis', [(0, '#e2a858'),
                 (1, '#990102')])


def rebase_angle(angle):
    """
    Swap axes for matplotlib.
    """
    return 0.5 * np.pi - angle


def draw_anisotropy(ax, objects_props, n_std=2):
    dist_ellipse_builder = DistributionEllipseBuilder()
    dist_ellipse_builder.build(objects_props)
    dist_ellipse = dist_ellipse_builder.distribution_ellipse
    dist_ellipse.full_theta = rebase_angle(dist_ellipse.full_theta)
    dist_ellipse.orientation = rebase_angle(dist_ellipse.orientation)
    r, theta, d = PolarPlots.sort_by_density(objects_props.axis_ratio,
                                             rebase_angle(objects_props.orientation))

    ax.scatter(theta, r, c=d, s=30, alpha=1, cmap='viridis')
    ax.plot(dist_ellipse.full_theta, dist_ellipse.full_radius, color='red')

    ax.quiver(0, 0, dist_ellipse.orientation, 0.5 * dist_ellipse.width,
              angles='xy', scale_units='xy', scale=1, color='red')
    ax.quiver(0, 0, 0.5 * np.pi + dist_ellipse.orientation,
              0.5 * dist_ellipse.height,
              angles='xy', scale_units='xy', scale=1, color='red')


def draw_segment(ax, image, objects_props):

    ax.imshow(image, cmap=cmap, origin='lower')

    _, _, density = PolarPlots.point_density(objects_props.axis_ratio,
                                             objects_props.orientation)

    density_cmap = colormaps.get_cmap('viridis')

    for i in range(len(objects_props['major_axis_length'])):
        color = density_cmap(density[i] / density.max())
        width = objects_props['major_axis_length'].iloc[i]
        height = objects_props['minor_axis_length'].iloc[i]
        alpha = objects_props['orientation'].iloc[i]
        xy = objects_props[['centroid-0', 'centroid-1']].iloc[i].to_numpy()

        res = PolarPlots.rotated_ellipse(width, height, rebase_angle(alpha))
        y, x = PolarPlots.polar_to_cartesian(*res)
        y += xy[1]
        x += xy[0]
        ax.plot(y, x, color=color, lw=1)

    ax.axis('off')


path = Path(__file__).parent.parent.parent.joinpath('data')
path_stats = Path(__file__).parent.parent.parent.joinpath('data')

heart = 'E11444_LMNA'
slice_name = 'E11444_08_SC2'

# heart = 'E11444_LMNA'
# slice_name = 'E11444_10_SC2'

n_radial_segments = 3
n_angular_segments = 36
node_step = 3

heart_slice_builder = HeartSliceBuilder()
heart_slice = heart_slice_builder.build_from_file(path, heart, slice_name,
                                                  n_angular_segments,
                                                  n_radial_segments, node_step)

path_slice_stats = path_stats.joinpath(heart, 'Stats', slice_name)
stats_loader = StatsLoader(path_stats)
object_stats = stats_loader.load_slice_data(path_slice_stats)

objects_props_builder = ObjectsPropertiesBuilder()
objects_props_builder.build_from_stats(object_stats, min_area=10)
objects_props_builder.add_slice_props(heart_slice)
objects_props = objects_props_builder.objects_props

centroids = objects_props[['centroid-0', 'centroid-1']].to_numpy(int)
object_mask = ((centroids[:, 0] > 2400) & (centroids[:, 0] < 2500))

object_mask &= ((centroids[:, 1] > 970) & (centroids[:, 1] < 1070))
objects_props = objects_props[object_mask]

# image = heart_slice.image[2400:2500, 970:1070]
# mask = image == 2

# mask = morphology.remove_small_objects(mask, 10)
# mask = segmentation.clear_border(mask)
# image[(mask == 0) & (image > 0)] = 1

n_std = 2

fig = plt.figure(figsize=(11, 5))
gs = fig.add_gridspec(1, 2, left=0.05, right=0.95,
                      bottom=0.05, top=0.9, wspace=0.05, hspace=0.2)
axs = []
axs.append(fig.add_subplot(gs[0]))
draw_segment(axs[0], heart_slice.image, objects_props)
axs[0].set_ylim(2400, 2500)
axs[0].set_xlim(970, 1070)

axs.append(fig.add_subplot(gs[1], projection='polar'))
draw_anisotropy(axs[-1], objects_props, n_std)

axs[0].set_title('A', loc='left', fontsize=14, y=1)
axs[1].set_title('B', loc='left', fontsize=14, y=0.985)
plt.show()

# fig.savefig('paperfigures/figures/segment_anisotropy.png', dpi=300,
#             bbox_inches='tight')
