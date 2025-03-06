from pathlib import Path
import numpy as np
import pyvista as pv

from skimage import io
from skimage.transform import pyramid_reduce
from matplotlib import colors
from fibrosisanalysis.parsers.image_loader import ImageLoader


cmap = colors.LinearSegmentedColormap.from_list(
    'fibrosis', [(0, 'white'),
                 (0.5, '#e2a858'),
                 (1, '#990102')])


class ImageParser:
    def __init__(self):
        pass

    @staticmethod
    def read_png(path):
        image = io.imread(path, as_gray=True)
        return np.add(image < 0.4, image < 0.8, dtype=int)

    @staticmethod
    def write_png(image, path):
        fib = [153, 1, 2]
        myo = [226, 168, 88]

        out_image = np.zeros((image.shape[0], image.shape[1], 3),
                             dtype=np.uint8)
        out_image[image == 1] = myo
        out_image[image == 2] = fib
        io.imsave(path, out_image)


path = Path('/Users/arstanbek/Library/CloudStorage/OneDrive-UGent/data')

heart = 'E11444_LMNA'
slice_name = 'E11444_08_SC2'

image_loader = ImageLoader(path)
max_shape = np.zeros(2)
images = []

for i in range(5, 10):
    slice_name = f'E11444_0{i}_SC2'
    image = image_loader.load_slice_data(path.joinpath(heart, 'DS',
                                                       slice_name))
    max_shape = np.maximum(max_shape, image.shape)
    images.append(image)


def reduce_image(img, scale=0.1):
    return pyramid_reduce(img, downscale=1/scale, preserve_range=True)


max_shape = np.zeros(2)

for i, img in enumerate(images):
    img = np.array(img)  # Ensure it's a NumPy array
    img = reduce_image(img, scale=0.5)  # Reduce image size
    images[i] = img
    max_shape = np.maximum(max_shape, img.shape)


points = []
values = []

for i, img in enumerate(images):
    image = np.zeros(max_shape.astype(int))
    j_min = int((max_shape[0] - img.shape[0]) // 2)
    j_max = j_min + img.shape[0]
    k_min = int((max_shape[1] - img.shape[1]) // 2)
    k_max = k_min + img.shape[1]
    image[j_min: j_max, k_min: k_max] = img
    points.append(np.column_stack((np.argwhere(image > 0),
                                   np.full(np.count_nonzero(image), 500 * i))))
    values.append(image[image > 0])

points = np.concatenate(points)
values = np.concatenate(values)

plotter = pv.Plotter()
# Create a PyVista point cloud
cloud = pv.PolyData(points)
# add scalars to the point cloud
cloud['Tissue'] = values
# plot the point cloud without scalar bar
plotter.add_points(cloud, scalars='Tissue', cmap=cmap, clim=[0, 2],
                   point_size=5, show_scalar_bar=False)
plotter.show(window_size=(2000, 1500), auto_close=False)
plotter.set_background('white', top='white')
plotter.screenshot('ga.png', transparent_background=True)
