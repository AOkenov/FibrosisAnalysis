from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from fibrosisanalysis.parsers.stats_loader import StatsLoader
from fibrosisanalysis.plots.point_density import PointDensity
from fibrosisanalysis.analysis.segment_properties import (
    StruturalAnisotrophy
)
from fibrosisanalysis.analysis.distribution_ellipses import (
    DistributionEllipses
)


# path = Path('./data').resolve()
path = Path('/Users/arstanbek/Projects/fibrosis-workspace/fibrosisanalysis/examples/data')
hearts = ['E10615_MYH7', 'E10621_ABCC9', 'E10691_RBM20', 'E10788_LMNA',
          'E10884', 'E10927_MYBPC3', 'E11442_TTN', 'E11443_LMNA',
          'E11444_LMNA', 'E11971_MYH7']

stats_loader = StatsLoader(path)
data = stats_loader.load_hearts_data(hearts)

print(data.head())

area_min = 100
area_max = 200
# position = 5

df = data.copy()
df = data[['area', 'minor_axis_length', 'major_axis_length', 'image', 
           'distance', 'orientation', 'tangent']]
df = df[df['area'].between(area_min, area_max)]
df['minor_axis_length'] = np.where(df['minor_axis_length'] == 0, 
                                   0.5, df['minor_axis_length'])
df['major_axis_length'] = np.where(df['major_axis_length'] == 0, 
                                   0.5, df['major_axis_length'])

df['position'] = np.digitize(df['distance'], np.linspace(0, 1.001, 4))

df['axis_ratio'] = df['major_axis_length'] / df['minor_axis_length']

df['theta'] = np.arcsin(np.sin(df['tangent'] - df['orientation']))
df['theta_x'] = df['axis_ratio'] * np.cos(df['theta'])
df['theta_y'] = df['axis_ratio'] * np.sin(df['theta'])

df = df[df['position'] == 2]

r = df['axis_ratio'].values
theta = df['theta'].values
theta_range = np.linspace(0, 2 * np.pi, 100)
n_std = 3

r = np.concatenate((r, r))
theta = np.concatenate((theta, theta + np.pi))
r, theta, density = PointDensity.polar(r, theta)

x, y = StruturalAnisotrophy.convert_to_cartesian(r, theta)
dist_ellipses = DistributionEllipses('error')
width, height, theta_ellipse = dist_ellipses.major_minor_axes(x, y, n_std)
ellipse_radius = PointDensity.rotated_ellipse(0.5 * width, 0.5 * height, 
                                              theta_range, theta_ellipse)

# r_ellipse, theta_ellipse = StruturalAnisotrophy.anysotrophy(x, y)

fig, axs = plt.subplots(1, 1, subplot_kw=dict(projection="polar"))
axs.scatter(theta, r, c=density, s=1)

axs.plot(theta_range, ellipse_radius,
         label='{} (n_std={})'.format('error', n_std), color='red')
axs.set_rmax(10)
plt.show()
