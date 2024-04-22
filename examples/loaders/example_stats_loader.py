from fibrosisanalysis.parsers.stats_loader import StatsLoader


path = '/Users/arstanbek/Hulk/Arstan/analysis/data'

hearts = ['E10615_MYH7', 'E10621_ABCC9', 'E10691_RBM20', 'E10788_LMNA',
          'E10884', 'E10927_MYBPC3', 'E11442_TTN', 'E11443_LMNA',
          'E11444_LMNA', 'E11971_MYH7']

stats_loader = StatsLoader(path)
# path = stats_loader.path.joinpath('E10615_MYH7', 'Stats', 'E10615_09_SC2_WABL')
# res = stats_loader.load_slice_data(path)
# res = stats_loader.load_heart_data(hearts[0])
res = stats_loader.load_hearts_data(hearts[:1])
print(res.info())
