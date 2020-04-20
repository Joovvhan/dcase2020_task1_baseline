hparams = dict()

hparams.update({
    'fs': 44100,
    'nsc_in_ms': 200,
    'mel_band_num': 80,
})

hparams.update({
    'nov_in_ms': int(hparams['nsc_in_ms'] / 2),
})

hparams.update({
    'nsc': int(hparams['fs'] * hparams['nsc_in_ms'] / 1000),
    'nov': int(hparams['fs'] * hparams['nov_in_ms'] / 1000)
})

hparams.update({
    'max_epoch': 1000,
    'batch_size': 256,
})

hparams.update({
    'logging_steps': 10,
})

# 10 Acoustic Scenes
SCENES = ('airport', 
          'shopping_mall', 
          'metro_station', 
          'street_pedestrian', 
          'public_square', 
          'street_traffic', 
          'tram',
          'bus',
          'metro',
          'park')

MAJOR_SCENES = ('indoor',
                'outdoor',
                'transportation')

# 15 Mobile Devices
DEVICES = ('a', 'b', 'c', 'd', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11')

# 12 Cities
CITIES = ('amsterdam', 'barcelona', 'helsinki', 'lisbon', 
          'london', 'lyon', 'madrid', 'milan', 
          'prague', 'paris', 'stockholm', 'vienna')

SCENE_DICT = {scene: i for i, scene in enumerate(SCENES)}
CITY_DICT = {city: i for i, city in enumerate(CITIES)}
DEVICE_DICT = {device: i for i, device in enumerate(DEVICES)}
M_SCENE_DICT = {m_scene: i for i, m_scene in enumerate(MAJOR_SCENES)}