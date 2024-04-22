from pathlib import Path
import matplotlib.pyplot as plt

from fibrosisanalysis.parsers.stats_loader import StatsLoader


path = Path(__file__).parent.parent.joinpath('data')

print(path)

# heart = 'E10691_RBM20'
# heart = 'E11444_LMNA'
# heart = 'E10927_MYBPC3'

hearts = ['E10615_MYH7', 'E10621_ABCC9', 'E10691_RBM20', 'E10788_LMNA',
          'E10884', 'E10927_MYBPC3', 'E11442_TTN', 'E11443_LMNA',
          'E11444_LMNA', 'E11971_MYH7']

objects_sizes = []
objects_densities = []
myocardiums = []

for heart in ['E11444_LMNA']:
    stats_loader = StatsLoader(path, subdir='Stats')
    res = stats_loader.load_hearts_data([heart])


mask = (res['solidity'] > 0.5) & (res['solidity'] < 0.6)
mask &= (res['area'] > 500) & (res['area'] < 700)
mask &= (res['major_axis_length'] / res['minor_axis_length'] > 5)
mask &= (res['major_axis_length'] / res['minor_axis_length'] < 10)
# mask &= (res['euler_number'] > -12) & (res['euler_number'] < -10)
print(res[mask][['solidity', 'area', 'major_axis_length', 'minor_axis_length',
                 'euler_number']].head())

res_0 = res[mask]

mask = (res['solidity'] > 0.7) & (res['solidity'] < 0.99)
mask &= (res['area'] > 500) & (res['area'] < 700)
# print(res[mask][['solidity', 'area', 'major_axis_length', 'minor_axis_length']])
mask &= (res['major_axis_length'] / res['minor_axis_length'] > 1)
mask &= (res['major_axis_length'] / res['minor_axis_length'] < 3)
# mask &= (res['euler_number'] > -12) & (res['euler_number'] < -10)
print(res[mask][['solidity', 'area', 'major_axis_length', 'minor_axis_length',
                 'euler_number']].head())

res_1 = res[mask]

# for i, image in enumerate(res_0['image']):
#     plt.imsave(f'0_{i}.png', image)

for i, image in enumerate(res_1['image']):
    plt.imsave(f'1_{i}.png', image)

# fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
# axs[0].imshow(res_0['image'].iloc[0])
# axs[1].imshow(res_1['image'].iloc[1])
# plt.show()
