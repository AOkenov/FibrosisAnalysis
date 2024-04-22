from pathlib import Path
from fibrosisanalysis.parsers.loader import Loader
from fibrosisanalysis.parsers.image_loader import ImageLoader
from fibrosisanalysis.parsers.edge_loader import EdgeLoader
from fibrosisanalysis.slice import HeartSliceBuilder
from fibrosisanalysis.segmentation import SplineEdge


class HeartSliceLoader(Loader):
    def __init__(self, path):
        super().__init__(path, subdir='', file_type='')
        self.image_loader = ImageLoader(path)
        self.edge_loader = EdgeLoader(path)
        self.heart_slice_builder = HeartSliceBuilder()

    def load_slice_data(self, path):
        image = self.image_loader.load_slice_data(path)
        edges = {}

        for i, name in enumerate(['epi', 'endo']):
            edge_path = path.with_name(path.stem + '_{}'.format(i))
            spline_edge = SplineEdge()
            spline_edge.nodes = self.edge_loader.load_slice_data()
            spline_edge.sample_nodes(30)

        edge_path = path.with_name(path.stem + '_')
        edges['endo'] = self.image_loader.load_slice_data(edge_path)
        edge_path = path.with_name(path.stem + '_1')
        edges['epi'] = self.image_loader.load_slice_data(edge_path)

        self.heart_slice_builder.reset()


path = Path('/Users/arstanbek/Hulk/Arstan/analysis/data')

hearts = ['E10615_MYH7', 'E10621_ABCC9', 'E10691_RBM20', 'E10788_LMNA',
          'E10884', 'E10927_MYBPC3', 'E11442_TTN', 'E11443_LMNA',
          'E11444_LMNA', 'E11971_MYH7']

image_loader = ImageLoader(path)
edge_loader = EdgeLoader(path)
image = image_loader.load_slice_data(path.joinpath('E10615_MYH7', 'Images',
                                                   'E10615_09_SC2_NABL'))
edges = {}

for i, name in enumerate(['epi', 'endo']):
    nodes = edge_loader.load_slice_data(path.joinpath('E10615_MYH7', 'Edges',
                                                      'E10615_09_SC2_WABL_{}'.format(i)))

    spline_edge = SplineEdge()
    spline_edge.nodes = nodes
    edges[name] = spline_edge.sample_nodes(30)
