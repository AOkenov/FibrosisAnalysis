from skimage import morphology, filters
from fibrosisanalysis.parsers.image_loader import ImageLoader


class DensityMap:
    def __init__(self, image, radius=50):
        self.radius = radius
        self.footprint = morphology.disk(self.radius)
        self.image = image

    def compute(self):
        fibrosis = (self.image == 2).astype('uint8') * 255
        self.density = filters.rank.mean(fibrosis, self.footprint,
                                         mask=self.image > 0)
        self.density[self.image == 0] = 0
        return self.density


class DensityMapBuilder:
    def __init__(self) -> None:
        pass

    def build_from_file(self, path, heart, slice_name, radius=50):
        image_loader = ImageLoader(path)
        image = image_loader.load_slice_data(path.joinpath(heart, 'Images',
                                                           slice_name))
        return DensityMap(image, radius)
