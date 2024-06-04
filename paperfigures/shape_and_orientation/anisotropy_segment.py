from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, colormaps
from skimage import morphology, segmentation
from sklearn.covariance import EmpiricalCovariance

from bitis.texture.properties import (
    DistributionEllipseBuilder,
    PolarPlots
)
from bitis.texture.analysis import ObjectAnalysis
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

    r = objects_props['axis_ratio'].values
    theta = swap_axis(objects_props['orientation'].values)
    r = np.concatenate([r, r])
    theta = np.concatenate([theta, theta + np.pi])

    r, theta, d = PolarPlots.sort_by_density(r, theta)
    ax.scatter(theta, r, c=d, s=30, alpha=1, cmap='viridis')

    dist_ellipse_builder = DistributionEllipseBuilder()
    dist_ellipse = dist_ellipse_builder.build(r, theta, n_std=n_std)
    orientation = dist_ellipse.orientation % np.pi
    ax.plot(dist_ellipse.full_theta, dist_ellipse.full_radius, color='red')
    ax.quiver(0, 0, orientation, 0.5 * dist_ellipse.width,
              angles='xy', scale_units='xy', scale=1, color='red')
    ax.quiver(0, 0, 0.5 * np.pi + orientation,
              0.5 * dist_ellipse.height,
              angles='xy', scale_units='xy', scale=1, color='red')
    
    # dist_ellipse_builder = DistributionEllipseBuilder()
    # dist_ellipse_builder.cov_estimator = EmpiricalCovariance()
    # dist_ellipse = dist_ellipse_builder.build(r, theta, n_std=n_std)
    # orientation = dist_ellipse.orientation
    # ax.plot(dist_ellipse.full_theta, dist_ellipse.full_radius, color='blue',
    #         ls='--')
    # # ax.quiver(0, 0, orientation, 0.5 * dist_ellipse.width,
    # #           angles='xy', scale_units='xy', scale=1, color='red')
    # # ax.quiver(0, 0, 0.5 * np.pi + orientation,
    # #           0.5 * dist_ellipse.height,
    # #           angles='xy', scale_units='xy', scale=1, color='red')




def draw_segment(ax, image, objects_props):
    ax.imshow(image, cmap=cmap, origin='lower', vmin=0, vmax=2)
    inds, density = PolarPlots.sorting_idx(objects_props['axis_ratio'].values,
                                           objects_props['orientation'].values)

    colors = colormaps.get_cmap('viridis')

    for i, dens in zip(inds, density):
        color = colors(dens / density.max())
        width = objects_props['major_axis_length'].iloc[i]
        height = objects_props['minor_axis_length'].iloc[i]
        alpha = objects_props['orientation'].iloc[i]
        centroids = objects_props[['centroid-0', 'centroid-1']].values
        xy = centroids[i]

        res = PolarPlots.rotated_ellipse(width, height, 0.5 * np.pi - alpha)
        y, x = PolarPlots.polar_to_cartesian(*res)
        y += xy[1]
        x += xy[0]
        ax.plot(y, x, color=color, lw=1)

    # ax.axis('off')


# path = Path(__file__).parent.parent.parent.joinpath('data')
path = Path('/Users/arstanbek/Library/CloudStorage/OneDrive-UGent/data')
path_stats = path

heart = 'E11444_LMNA'
slice_name = 'E11444_08_SC2'

image_loader = ImageLoader()
image = image_loader.load_slice_data(path.joinpath(heart, 'Images',
                                                   slice_name))

image = image[2400:2500, 970:1070]
mask = image == 2

mask = morphology.remove_small_objects(mask, 10)
mask = segmentation.clear_border(mask)
image[(mask == 0) & (image > 0)] = 1
image[mask == 0] = 1

objects_analysis = ObjectAnalysis()
objects_props = objects_analysis.build_props(mask)

n_std = 2

fig = plt.figure(figsize=(8, 4))
gs = fig.add_gridspec(1, 2, left=0.05, right=0.95,
                      bottom=0.1, top=0.8, wspace=0.1, hspace=0.2)
axs = []
axs.append(fig.add_subplot(gs[0]))
draw_segment(axs[0], image, objects_props)

axs.append(fig.add_subplot(gs[1], projection='polar'))
draw_anisotropy(axs[-1], objects_props, n_std)

axs[0].set_title('A. Fibrotic Clusters', loc='left', fontsize=12, y=1.1)
axs[1].set_title('B. Structural Anisotropy', loc='left', fontsize=12, y=1.1)
plt.show()

# fig.savefig('paperfigures/figures/segment_anisotropy.png', dpi=300,
#             bbox_inches='tight')
