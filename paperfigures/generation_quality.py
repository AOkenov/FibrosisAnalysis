from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from bitis.texture.texture import Texture
from bitis.texture.properties import (
    PatternPropertiesBuilder,
    DistributionEllipseBuilder,
    PolarPlots
)


def swap_axis(angle):
    """
    Swap axis for polar plot.
    """
    return 0.5 * np.pi - angle


def draw_anisotropy(ax, objects_props, n_std=2):
    objects_props = objects_props[objects_props['area'] >= 5]
    r = objects_props['axis_ratio'].values
    theta = objects_props['orientation'].values

    r = np.concatenate([r, r])
    theta = np.concatenate([theta, theta + np.pi])

    theta = swap_axis(theta)

    dist_ellipse_builder = DistributionEllipseBuilder()
    dist_ellipse = dist_ellipse_builder.build(r, theta, n_std=n_std)
    full_theta = dist_ellipse.full_theta
    orientation = dist_ellipse.orientation
    r, theta, d = PolarPlots.sort_by_density(r, theta)

    ax.scatter(theta, r, c=d, s=30, alpha=1, cmap='viridis')
    ax.plot(full_theta, dist_ellipse.full_radius, color='red')

    ax.quiver(0, 0, orientation, 0.5 * dist_ellipse.width,
              angles='xy', scale_units='xy', scale=1, color='red')
    ax.quiver(0, 0, 0.5 * np.pi + orientation,
              0.5 * dist_ellipse.height,
              angles='xy', scale_units='xy', scale=1, color='red')
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelsize=10)


def calc_ccdf(df):
    count = np.bincount(df['area'].values.astype(int))
    area_bins = np.arange(1 + df['area'].max())

    area_bins = area_bins[1:]
    count = count[1:]
    ccdf = np.cumsum(count[::-1])[::-1] / np.sum(count)
    return area_bins, ccdf


def calc_area_cdf(df):
    count = np.bincount(df['area'].values.astype(int))
    area_bins = np.arange(1 + df['area'].max())

    area_bins = area_bins[1:]
    count = count[1:]
    area = area_bins * count

    cdf = np.cumsum(area) / np.sum(area)
    return area_bins, cdf


def draw_area_cdf(ax, objects_props, label=''):
    area_bins, cdf = calc_area_cdf(objects_props)
    ax.plot(area_bins, cdf, label=label)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
    ax.set_xlabel('Size')
    ax.set_ylabel('Fibrotic Tissue')
    ax.set_xscale('log')


def draw_ccdf(ax, objects_props, label=''):
    area_bins, ccdf = calc_ccdf(objects_props)
    ax.plot(area_bins, ccdf, label=label)
    ax.set_xlabel('Size')
    ax.set_ylabel('CCDF')
    ax.set_yscale('log')
    ax.set_xscale('log')


def draw_perimeter_cdf(ax, objects_props, label=''):
    count = np.bincount(objects_props['area'].values,
                        weights=objects_props['perimeter'].values)
    area_bins = np.arange(1 + objects_props['area'].max())

    area_bins = area_bins[1:]
    count = count[1:]

    cdf = np.cumsum(count)

    ax.plot(area_bins, cdf, label=label)
    ax.set_xlabel('Size')
    ax.set_ylabel('Perimeter')
    ax.set_xscale('log')
    ax.set_yscale('log')


path = Path(__file__).parents[1].joinpath('data_gen_paper')

fibrosis_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'fibrosis', [(0, 'white'),
                 (0.5, '#e2a858'),
                 (1, '#990102')])

files = path.joinpath('original').glob('*.npy')
files = [f.name for f in files]

for file_name in files:
    nim_gen = np.load(path.joinpath('generated', file_name))
    nim = np.load(path.joinpath('original', file_name))

    nim_uni = np.random.random(nim_gen.shape) < (np.sum(nim == 2) / nim.size)
    nim_uni = nim_uni.astype(int) + 1

    textures = []
    pattern_props = []

    for im in [nim, nim_gen, nim_uni]:
        # im = ErosionSegmentation.segment(im == 2)
        pattern_builder = PatternPropertiesBuilder(area_quantile=0.95)
        pattern_properties = pattern_builder.build(im == 2)
        pattern_props.append(pattern_properties)

        texture = Texture()
        texture.matrix = im
        texture.properties["pattern"] = pattern_properties
        texture.properties["object_props"] = pattern_builder.object_props
        textures.append(texture)

    print(pd.concat(pattern_props))

    fig, axs = plt.subplot_mosaic([['im_or', 'im_or',
                                    'im_gen', 'im_gen',
                                    'im_uni', 'im_uni'],
                                   ['sa', 'sa',
                                    'sa_gen', 'sa_gen',
                                       'sa_uni', 'sa_uni'],
                                   ['ccdf', 'ccdf',
                                    'ccdf', 'cdf_area',
                                    'cdf_area', 'cdf_area']],
                                  per_subplot_kw={('sa', 'sa_gen', 'sa_uni'
                                                   ): {'projection': 'polar'}},
                                  height_ratios=[2, 1, 1],
                                  figsize=(8, 7))

    axs['im_gen'].sharex(axs['im_or'])
    axs['im_gen'].sharey(axs['im_or'])

    axs['im_uni'].sharex(axs['im_or'])
    axs['im_uni'].sharey(axs['im_or'])

    axs['sa'].sharex(axs['sa_gen'])
    axs['sa'].sharey(axs['sa_gen'])

    axs['sa_uni'].sharex(axs['sa_gen'])
    axs['sa_uni'].sharey(axs['sa_gen'])

    # axs['ccdf'].sharex(axs['cdf_area'])
    # axs['ccdf'].sharey(axs['cdf_area'])
    # ax'].sharex(axs['cdf_area'])
    # ax'].sharey(axs['cdf_area'])

    draw_anisotropy(axs['sa'], textures[0].properties["object_props"])
    draw_anisotropy(axs['sa_gen'], textures[1].properties["object_props"])
    draw_anisotropy(axs['sa_uni'], textures[2].properties["object_props"])

    draw_ccdf(axs['cdf_area'],
              textures[0].properties["object_props"],
              label='Original')
    draw_ccdf(axs['cdf_area'],
              textures[1].properties["object_props"],
              label='DS Generator')
    draw_ccdf(axs['cdf_area'],
              textures[2].properties["object_props"],
              label='Uniform Generator')

    draw_area_cdf(axs['ccdf'],
                  textures[0].properties["object_props"],
                  label='Original')
    draw_area_cdf(axs['ccdf'],
                  textures[1].properties["object_props"],
                  label='DS Generator')
    draw_area_cdf(axs['ccdf'],
                  textures[2].properties["object_props"],
                  label='Uniform Generator')

    axs['cdf_area'].legend(fontsize=8, loc='lower left')
    axs['ccdf'].legend(fontsize=8, loc='lower left')

    axs['im_or'].imshow(textures[0].matrix, origin='lower', vmin=0, vmax=2,
                        cmap=fibrosis_cmap)
    axs['im_or'].set_title('Original Texture')

    axs['im_gen'].imshow(textures[1].matrix, origin='lower', vmin=0, vmax=2,
                         cmap=fibrosis_cmap)
    axs['im_gen'].set_title("DS Generator")

    axs['im_uni'].imshow(textures[2].matrix, origin='lower', vmin=0, vmax=2,
                         cmap=fibrosis_cmap)
    axs['im_uni'].set_title("Uniform Generator")
    # plt.subplots_adjust(hspace=0.5, wspace=0.6)
    plt.tight_layout()
    plt.show()

    fig.savefig(Path(__file__).parent.joinpath('figures',
                                               file_name.replace('.npy',
                                                                 '.png')),
                dpi=300)
