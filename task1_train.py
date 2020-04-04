import os
import csv
from collections import namedtuple, defaultdict
import librosa
from copy import deepcopy
import queue
import random
from threading import Thread
from tqdm import tqdm
import numpy as np


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

from hparams import hparams, DEVICE_DICT, CITY_DICT, SCENE_DICT

DataWithLabel = namedtuple('DataWithLabel', ['mel_spectrogram', 'file_path', 'city', 'device', 'scene'])
DataBatch = namedtuple('DataBatch', ['mel_spectrogram_batch', 'file_path_list', 'city_label', 'device_label', 'scene_label'])
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
    mel_spectrogram = librosa.feature.melspectrogram(y, sr=sr, n_fft=hparams['nsc'], hop_length=hparams['nov'], n_mels=hparams['mel_band_num'])
    return mel_spectrogram

def data_list_to_batch(data_list: [DataWithLabel]):

    assert len(data_list) > 0, " * Empty Data List"

    batch_size = len(data_list)
    sample_mel_spectrogram = getattr(data_list[0], 'mel_spectrogram')
    num_mels, T = sample_mel_spectrogram.shape

    batch_mel_spectrogram = np.zeros([batch_size, num_mels, T])
    city_label = np.ones(batch_size) * -1
    scene_label = np.ones(batch_size) * -1
    device_label = np.ones(batch_size) * -1
    file_path_list = list()

    for i, data_with_label in enumerate(data_list):
        mel_spectrogram = getattr(data_with_label, 'mel_spectrogram')
        file_path = getattr(data_with_label, 'file_path')
        city = getattr(data_with_label, 'city')
        device = getattr(data_with_label, 'device')
        scene = getattr(data_with_label, 'scene')

        batch_mel_spectrogram[i] = mel_spectrogram
        city_label[i] = CITY_DICT[city]
        scene_label[i] = SCENE_DICT[scene]
        device_label[i] = DEVICE_DICT[device]
        file_path_list.append(file_path)

    return DataBatch(torch.Tensor(batch_mel_spectrogram), file_path_list, city_label, device_label, scene_label)

class LoadingThread(Thread):
    def __init__(self, metadata, queue):
        Thread.__init__(self)
        self.metadata = metadata
        self.queue = queue
    
    def run(self):
        for file_info in self.metadata:
            mel_spectrogram = load_mel_spectrogram(getattr(file_info, 'file_path'))
            self.queue.put(DataWithLabel(mel_spectrogram, *file_info))

class DatasetLoader:

    def __init__(self, metadata, num_thread=2):
        self.metadata = deepcopy(metadata)
        self.queue = queue.Queue(maxsize=hparams['batch_size'] * 5)
        self.num_thread = num_thread
        self.thread_list = list()
        self.max_size = len(self.metadata)
        self.batch_size = hparams['batch_size']
        self.max_batch_size = int(np.ceil(self.max_size / self.batch_size))

        self.num_loaded_files = 0

    def shuffle_metadata(self):
        random.shuffle(self.metadata)

    def __repr__(self):
        return '[ {}, ..., {} ]'.format(self.metadata[0], self.metadata[-1])

    def start_loading(self):
        self.shuffle_metadata()
        self.thread_list = list()
        self.num_loaded_files = 0

        for i in range(self.num_thread):
            t = LoadingThread(self.metadata[i::self.num_thread], self.queue)
            t.start()
            self.thread_list.append(t)

        print('[ Start Loading ]')

    def get_queue(self):
        if self.num_loaded_files < self.max_size:
            content = self.queue.get()
            self.num_loaded_files += 1
            return content
        else:
            # print(' * End of queue reached')
            return None
        

    def generator(self):
        # print('[Inside Generator]')
        while any(map(Thread.is_alive, self.thread_list)):

            batch_data_list = list()

            while len(batch_data_list) < self.batch_size:
                data_with_label = self.get_queue()
                if data_with_label is not None:
                    batch_data_list.append(data_with_label)
                else:
                    break

            if len(batch_data_list) > 0:
                yield data_list_to_batch(batch_data_list)

def train(**argv):
    metadata_path = argv['metadata_path']
    print(argv)

class CRNN_Net(nn.Module):
    def __init__(self):
        super(CRNN_Net, self).__init__()
        self.conv = nn.Conv2d(1, 256, kernel_size=(80, 11), stride=5)
        self.rnn_1 = nn.RNN(256, 256)
        self.rnn_2 = nn.RNN(256, 128)
#         self.rnn_3 = nn.RNN(128, 64)
#         self.rnn_4 = nn.RNN(64, 32)
        self.fc = nn.Linear(128 * 19, len(SCENE_DICT))
        # (B, M[80], H[1], T[101]) => (B, M[80], H[256], T[21])
        
    def forward(self, tensor):
        tensor = tensor.unsqueeze(1)
        tensor = F.relu(self.conv(tensor))
        tensor = tensor.squeeze(2)
        tensor = tensor.permute(2, 0, 1) # (B, H, T) => (T, B, H)
        tensor, _ = self.rnn_1(tensor)
        tensor, _ = self.rnn_2(tensor)
#         tensor, _ = self.rnn_3(tensor)
#         tensor, _ = self.rnn_4(tensor)
        tensor = tensor.permute(1, 2, 0) # (T, B, H) => (B, H, T)
        tensor = tensor.reshape(tensor.shape[0], -1)
        tensor = self.fc(tensor)
        tensor = F.log_softmax(tensor, dim=-1)

        return tensor

net = CRNN_Net()

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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = CRNN_Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(hparams['max_epoch']):

        print('[ Training Epoch #{:03d} ]'.format(epoch))
        train_dataset_loader.start_loading()
        for i, batch in tqdm(enumerate(train_dataset_loader.generator()), total=train_dataset_loader.max_batch_size):
            
            mel_spectrogram_batch, file_path_list, city_label, device_label, scene_label = batch
            optimizer.zero_grad()
            outputs = net(mel_spectrogram_batch.to(device))
            loss = criterion(outputs, torch.tensor(scene_label, dtype=torch.int64).to(device))
            loss.backward()
            optimizer.step()

            print(loss.item())

            # print(batch[0].shape)

        print('[ Evaluation Epoch #{:03d} ]'.format(epoch))
        eval_dataset_loader.start_loading()
        for i, batch in tqdm(enumerate(eval_dataset_loader.generator()), total=eval_dataset_loader.max_batch_size):
            pass
            # print(i, batch)
            # print(batch[0].shape)

    # inspect_metadata(metadata_train)

    # inspect_metadata_audio_file(metadata_eval)

    # inspect_metadata_audio_file(dataset_base_path, metadata_train)

    '''
        
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    '''