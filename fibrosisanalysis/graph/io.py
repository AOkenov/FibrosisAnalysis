import json
import PIL
import numpy as np


class DataEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(DataEncoder, self).default(obj)


class IO:
    def __init__(self):
        pass

    @staticmethod
    def save_graph_as_json(path, dirname, filename, data):
        with open(path + dirname + '{}.json'.format(filename), 'w') as f:
            json.dump(data, f, cls=DataEncoder)

    @staticmethod
    def load_json(path, dirname, filename, data):
        with open(path + dirname + '{}.json'.format(filename), 'r') as f:
            data = json.load(f)
        return data

    @staticmethod
    def save_image_as_jpg(path, dirname, filename, image):
        im = PIL.Image.fromarray(image)
        im.save(path + dirname + '{}.jpg'.format(filename))
