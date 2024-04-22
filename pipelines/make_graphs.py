import sys
sys.path.append('/mnt/8TBdrive/Arstan/')
sys.path.append('/mnt/8TBdrive/Arstan/Finitewave/')

import os
import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm
from fibrosisprocessing.graph import Image, Filters
from fibrosisprocessing.graph import IO as GraphIO
from fibrosisprocessing.tools import IO as DataFrameIO


DIRNAME = 'StatsGenerated/'
N = 1000

path = '/mnt/8TBdrive/Arstan/analysis/data/'
path_save = path
dirs = ['E10615_MYH7/', 'E10621_ABCC9/', 'E10691_RBM20/', 'E10788_LMNA/',
        'E10884/', 'E10927_MYBPC3/', 'E11442_TTN/', 'E11443_LMNA/',
        'E11444_LMNA/', 'E11971_MYH7/']


def get_files(path, dirname):
    path = path + dirname
    files = [f.split('_0.npy')[0] for f in os.listdir(path + 'Edges/')
             if '_0.npy' in f]
    return sorted(files)


def load_df(path, dirs):
    ddf = []
    for dirname in dirs:
        files = get_files(path, dirname)
        ddf.append(DataFrameIO.load_df_from_list(path + dirname, DIRNAME,
                                                 files))
    return pd.concat(ddf, ignore_index=True)


def mean_width(data, medial_axis, coords):
    return np.mean(medial_axis[tuple(coords[data['route']].T)])


def create_graph(path_save, image, ind):
    image = Image.mask(image)
    skelet = Image.skelet(image)
    graph = Image.graph(skelet)
    medial_axis = Image.medial_axis(image, skelet)
    coords = Image.coords(skelet)

    graph = Filters.remove_diagonal_edges(graph)
    graph = Filters.add_node_attributes(graph, coords, 'pos')
    graph_reduced = Filters.reduce(graph)

    graph_reduced = Filters.add_edge_attributes(graph_reduced, mean_width,
                                                'width', medial_axis, coords)

    data = nx.node_link_data(graph_reduced)
    [d.pop('route') for d in data['links']]

    # GraphIO.save_image_as_jpg(path_save, 'images/', ind, image)
    GraphIO.save_graph_as_json(path_save, 'graphs/', ind, data)


df = load_df(path, dirs[:1])
mask = df['area'].between(1e2, 5e2)
df = df[['density', 'image']][mask]

bar = tqdm(total=N)

inds = np.random.choice(np.arange(df.shape[0]), size=min(N, df.shape[0]))

df = df.iloc[inds]
df = df.reset_index(drop=True)
for index, row in df.iterrows():
    create_graph(path_save, row['image'], index)
    df.loc[index, 'filename'] = '{}.json'.format(index)
    bar.update(1)

bar.close()

DataFrameIO.save_df(path_save, '', 'graphs_props', df)
