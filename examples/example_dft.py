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


cmap = colors.LinearSegmentedColormap.from_list(
    'fibrosis', [(0, 'white'),
                 (0.5, '#e2a858'),
                 (1, '#990102')])

cmap_2 = colors.LinearSegmentedColormap.from_list(
    'fibrosis', [(0, '#e2a858'),
                 (1, '#990102')])


def draw_anisotropy(ax, objects_props, n_std=2):
    dist_ellipse_builder = DistributionEllipseBuilder()
    dist_ellipse_builder.build(objects_props)
    dist_ellipse = dist_ellipse_builder.distribution_ellipse

    r, theta, d = PolarPlots.sort_by_density(objects_props.axis_ratio,
                                             objects_props.orientation)

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

    for i in range(len(objects_props.major_axis)):
        color = density_cmap(density[i] / density.max())
        width = objects_props.major_axis[i]
        height = objects_props.minor_axis[i]
        alpha = objects_props.orientation[i]
        xy = objects_props.centroids[i]

        res = PolarPlots.rotated_ellipse(width, height, alpha)
        y, x = PolarPlots.polar_to_cartesian(*res)
        y += xy[1]
        x += xy[0]
        ax.plot(y, x, color=color, lw=1)

    ax.axis('off')


path = Path('/Users/arstanbek/Hulk/Arstan/analysis/data')
path_stats = Path(
    '/Users/arstanbek/Projects/fibrosis-workspace/fibrosisanalysis/examples/data')

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

image = heart_slice.image[2400:2500, 970:1070]
mask = image == 2

fft = np.fft.fft2(mask.astype(float))
# fshift = np.fft.fftshift(fft)

magnitude = np.abs(fft)
angle = np.angle(fft)

angle = angle[magnitude < 500]
magnitude = magnitude[magnitude < 500]

magnitude, angle, density = PolarPlots.sort_by_density(magnitude, angle)

fig, axs = plt.subplots(subplot_kw={'projection': 'polar'})
axs.scatter(angle, magnitude, c=density, s=30, alpha=1, cmap='viridis')
plt.show()
