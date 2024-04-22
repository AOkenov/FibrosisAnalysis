from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from fibrosisanalysis.parsers.edge_loader import EdgeLoader
from fibrosisanalysis.segmentation import (
    SplineEdge
)

path = Path('/Users/arstanbek/Hulk/Arstan/analysis/data')

hearts = ['E10615_MYH7', 'E10621_ABCC9', 'E10691_RBM20', 'E10788_LMNA',
          'E10884', 'E10927_MYBPC3', 'E11442_TTN', 'E11443_LMNA',
          'E11444_LMNA', 'E11971_MYH7']

loader = EdgeLoader(path)

edges = {}

for i, name in enumerate(['epi', 'endo']):
    nodes = loader.load_slice_data(path.joinpath('E10615_MYH7', 'Edges',
                                                 'E10615_09_SC2_WABL_{}'.format(i)))

    spline_edge = SplineEdge()
    spline_edge.nodes = nodes
    edges[name] = spline_edge.sample_nodes(30)

# res = loader.load_heart_data(hearts[0])
# res = loader.load_hearts_data(hearts[:1])
# print(res.head())

plt.figure()
for name, edge in edges.items():
    plt.plot(edge[:, 0], edge[:, 1], marker='o', label=name)
plt.legend()
plt.show()
