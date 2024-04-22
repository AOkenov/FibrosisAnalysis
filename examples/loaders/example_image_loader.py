from pathlib import Path
import matplotlib.pyplot as plt

from fibrosisanalysis.parsers.image_loader import ImageLoader
from fibrosisanalysis.segmentation import (
    SplineEdge
)

path = Path('/Users/arstanbek/Hulk/Arstan/analysis/data')

hearts = ['E10615_MYH7', 'E10621_ABCC9', 'E10691_RBM20', 'E10788_LMNA',
          'E10884', 'E10927_MYBPC3', 'E11442_TTN', 'E11443_LMNA',
          'E11444_LMNA', 'E11971_MYH7']

loader = ImageLoader(path)
image = loader.load_slice_data(path.joinpath('E10615_MYH7', 'Images',
                                             'E10615_09_SC2_NABL'))

# res = loader.load_heart_data(hearts[0])
# res = loader.load_hearts_data(hearts[:1])
# image = res[(res['Slice'] == 'E10615_09_SC2_NABL')]['Image'][0]

print(image.shape)

binary = image == 2

plt.figure()
plt.imshow(binary, origin='lower')
plt.show()
