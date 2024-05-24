import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from fibrosisanalysis.analysis.distribution_ellipses import (
    DistributionEllipses
)
from fibrosisanalysis.analysis.segments_properties import (
    StruturalAnisotrophy
)
from fibrosisanalysis.plots.point_density import PointDensity


def rotated_ellipse(a, b, theta, alpha):
    return (a * b) / np.sqrt((b * np.cos(theta - alpha))**2
                             + (a * np.sin(theta - alpha))**2)


def points_density(r, theta):
    xy = np.vstack([np.sin(theta) * r, np.cos(theta) * r])
    d = stats.gaussian_kde(xy)(xy)
    idx = d.argsort()
    return r[idx], theta[idx], d[idx]


cov = np.array([[3, 2], [2, 3]])
points = np.random.multivariate_normal(mean=(0, 0), cov=cov, size=10000)
mean_pos = points.mean(axis=0)

x, y = points.T

colors = ['b', 'r']
n_std = 1

theta_range = np.linspace(0, 2 * np.pi, 100)

r, theta = StruturalAnisotrophy.convert_to_polar(x, y)

r, theta, density = PointDensity.polar(r, theta)

fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))

ax.scatter(theta, r, c=density, s=1)
ax.set_rmax(10)

for i, ellipse_type in enumerate(['error', 'confidence']):
    dist_ellipses = DistributionEllipses(ellipse_type)
    width, height, theta = dist_ellipses.major_minor_axes(x, y, n_std)

    U = theta
    V = 0.5 * width

    print(width / height)

    ax.quiver(0, 0, U, V, angles='xy', scale_units='xy', scale=1,
              color=colors[i])

    U = 0.5 * np.pi + theta
    V = 0.5 * height

    ax.quiver(0, 0, U, V, angles='xy', scale_units='xy', scale=1,
              color=colors[i])

    ellipse_radius = PointDensity.rotated_ellipse(0.5 * width, 0.5 * height, 
                                                  theta_range, theta)

    ax.plot(theta_range, ellipse_radius,
            label='{} (n_std={})'.format(ellipse_type, n_std),
            color=colors[i])
plt.legend()
plt.show()
