import os
import csv
from collections import namedtuple, defaultdict
import librosa
from copy import deepcopy
import queue
import random
from threading import Thread
from tqdm import tqdm

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
    'max_epoch': 2,
})

scene_dict = dict()
city_dict = dict()
device_dict = dict()

BatchData = namedtuple('BatchData', ['mel_spectrogram', 'file_path', 'city', 'device', 'scene'])
FileInfo = namedtuple('FileInfo', ['file_path', 'city', 'device', 'scene'])

def load_metadata(metadata_path, base_dataset_path):

    with open(metadata_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter='\t')

        next(csv_reader)

        if 'test.csv' not in metadata_path:
            metadata = [FileInfo(os.path.join(base_dataset_path, line[0]), line[0].split('-')[1], line[0].split('-')[-1].replace('.wav', ''), line[1]) 
                        for line in csv_reader]
        else:
            metadata = [FileInfo(os.path.join(base_dataset_path, line[0]), line[0].split('-')[1], line[0].split('-')[-1].replace('.wav', ''), 'UNKNOWN') 
                        for line in csv_reader]

    print('[ Loaded {} ] [ Length: {} ]'.format(metadata_path, len(metadata)))
    
    return metadata

def display_dict(dict_obejct, display_sum=False):

    print()

    sum = 0

    for key in list(dict_obejct.keys()):
        print('{:>20} | {:<20}'.format(key, dict_obejct[key]))
        sum += dict_obejct[key]
    
    if display_sum:
        print('{:>20} | {:<20}'.format('SUM', sum))
    print()

def inspect_metadata(metadata):

    city_counter = defaultdict(int)
    scene_counter = defaultdict(int)
    device_counter = defaultdict(int)

    for file_info in metadata:
        scene_counter[getattr(file_info, 'scene')] += 1
        city_counter[getattr(file_info, 'city')] += 1  
        device_counter[getattr(file_info, 'device')] += 1  

    display_dict(city_counter)
    display_dict(scene_counter)
    display_dict(device_counter)

def inspect_metadata_audio_file(metadata):

    for file_info in metadata:
        file_path = getattr(file_info, 'file_path')
        duration = librosa.core.get_duration(filename=file_path)
        fs = librosa.core.get_samplerate(file_path)
        print('[ File: {:<50s} ] [ Duration: {:05.2f} ] [ Sampling Rate: {:<6d} ] '.format(file_path, duration, fs))

def load_mel_spectrogram(file_path):
    y, sr = librosa.core.load(file_path, sr=hparams['fs'])
    mel_spectrogram = librosa.feature.melspectrogram(y, sr=sr, n_fft=hparams['nsc'], hop_length=hparams['nov'])
    return mel_spectrogram

class LoadingThread(Thread):
    def __init__(self, metadata, queue):
        Thread.__init__(self)
        self.metadata = metadata
        self.queue = queue
    
    def run(self):
        for file_info in self.metadata:
            mel_spectrogram = load_mel_spectrogram(getattr(file_info, 'file_path'))
            self.queue.put(BatchData(mel_spectrogram, *file_info))

class DatasetLoader:

    def __init__(self, metadata, num_thread=2):
        self.metadata = deepcopy(metadata)
        self.queue = queue.Queue(maxsize=10)
        self.num_thread = num_thread
        self.thread_list = list()
        self.max_size = len(self.metadata)

    def shuffle_metadata(self):
        random.shuffle(self.metadata)

    def __repr__(self):
        return '[ {}, ..., {} ]'.format(self.metadata[0], self.metadata[-1])

    def start_loading(self):
        self.shuffle_metadata()
        self.thread_list = list()

        for i in range(self.num_thread):
            t = LoadingThread(self.metadata[i::self.num_thread], self.queue)
            t.start()
            self.thread_list.append(t)

        print('[ Start Loading ]')
        

    def generator(self):
        # print('[Inside Generator]')
        while any(map(Thread.is_alive, self.thread_list)) or not self.queue.empty():
            batch_data = self.queue.get()
            yield batch_data


def train(**argv):
    metadata_path = argv['metadata_path']
    print(argv)


if __name__ == '__main__':

    print('{:^43s}'.format('==== Display hparams ===='))
    display_dict(hparams)
    
    dataset_name = 'TAU-urban-acoustic-scenes-2020-mobile-development'

    dataset_base_path = os.path.join('datasets', dataset_name)
    
    metadata_train_path = dataset_base_path + '/' + 'evaluation_setup' + '/' + 'fold1_train.csv'

    # metadata_test_path = 'datasets/' + dataset_name + '/' + 'evaluation_setup' + '/' + 'fold1_test.csv' 

    metadata_eval_path = dataset_base_path + '/' + 'evaluation_setup' + '/' + 'fold1_evaluate.csv' 

    metadata_train = load_metadata(metadata_train_path, dataset_base_path)

    # metadata_test = load_metadata(metadata_path=metadata_test_path)

    metadata_eval = load_metadata(metadata_eval_path, dataset_base_path)

    train_dataset_loader = DatasetLoader(metadata_train)
    eval_dataset_loader = DatasetLoader(metadata_eval)

    print(train_dataset_loader)

    for epoch in range(hparams['max_epoch']):

        print('[ Training Epoch #{:03d} ]'.format(epoch))
        train_dataset_loader.start_loading()
        for i, batch in tqdm(enumerate(train_dataset_loader.generator()), total=train_dataset_loader.max_size):
            # print(i, batch)
            pass

        print('[ Evaluation Epoch #{:03d} ]'.format(epoch))
        eval_dataset_loader.start_loading()
        for i, batch in tqdm(enumerate(eval_dataset_loader.generator()), total=eval_dataset_loader.max_size):
            # print(i, batch)
            pass

    # inspect_metadata(metadata_train)

    # inspect_metadata_audio_file(metadata_eval)

    # inspect_metadata_audio_file(dataset_base_path, metadata_train)