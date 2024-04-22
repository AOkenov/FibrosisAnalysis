from .image_loader import ImageLoader


class DensityLoader(ImageLoader):
    def __init__(self, path=''):
        super().__init__(path)

    def rescale(self, image):
        return image / 255
