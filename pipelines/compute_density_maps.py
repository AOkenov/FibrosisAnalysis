from pathlib import Path
from tqdm import tqdm
import numpy as np
from fibrosisanalysis.parsers.image_loader import ImageLoader


path = Path(__file__).parent.parent.joinpath('data')

hearts = ['E10615_MYH7', 'E10621_ABCC9', 'E10691_RBM20', 'E10788_LMNA',
          'E10884', 'E10927_MYBPC3', 'E11442_TTN', 'E11443_LMNA',
          'E11444_LMNA', 'E11971_MYH7']


image_loader = ImageLoader(path)

for heart in hearts:
    path_image = path.joinpath(heart, 'Images')
    path_density = path.joinpath(heart, 'Density')
    files = list(path_image.glob('*{}'.format('.png')))
    files = [file.stem for file in files if not file.name.startswith('.')]

    with tqdm(total=len(files)) as pbar:
        for file in files:
            image = image_loader.load_png(path_image.joinpath(file))
            density_map = image_loader.load_png(path_density.joinpath(file))
            if np.any(density_map.shape != image.shape):
                print(file)
                print(density_map.shape, image.shape)
            pbar.update()
