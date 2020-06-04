import os
import csv
import datetime
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

from torch.utils.tensorboard import SummaryWriter

from hparams import hparams, DEVICE_DICT, CITY_DICT, SCENE_DICT

hparams.update({
    'mel_band_num': 81,
})

# hparams.update({
#     'fs': 48000,
# })

# hparams.update({
#     'nsc': int(hparams['fs'] * hparams['nsc_in_ms'] / 1000),
#     'nov': int(hparams['fs'] * hparams['nov_in_ms'] / 1000)
# })

# hparams.update({
#     'batch_size': 64,
# })

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

class Averager:
    
    def __init__(self, value = 0, count = 0):
        self.value = value
        self.count = count
        
    def __iadd__(self, value):
        self.value = (self.count * self.value + value) / (self.count + 1)
        self.count += 1
        
        return self
    
    def __add__(self, value):
        new_value = (self.count * self.value + value) / (self.count + 1)
        new_count = self.count + 1
        new_averager = Averager(new_value, new_count)
        return new_averager
    
    def __repr__(self):
        return 'Value: {} / Count: {}'.format(self.value, self.count)

    def reset(self):
        self.value = 0
        self.count = 0


def train(**argv):
    metadata_path = argv['metadata_path']
    print(argv)


class CRNN_Net_2(nn.Module):
    
    def __init__(self):
        super(CRNN_Net_2, self).__init__()
        # self.conv = nn.Conv2d(1, 256, kernel_size=(80, 11), stride=5)
        # self.rnn_1 = nn.RNN(256, 256)
        # self.rnn_2 = nn.RNN(256, 128)

        self.ex_conv = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.ex_conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.ex_conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.ex_conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)

        self.conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=3)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=3)
        # self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=3)

        self.conv_layers = [self.ex_conv, self.ex_conv2, self.ex_conv3, self.ex_conv4,
            self.conv, self.conv2, self.conv3, self.conv4]

        self.batch_norm_layers = nn.ModuleList([nn.BatchNorm2d(16), 
            nn.BatchNorm2d(32), 
            nn.BatchNorm2d(64), 
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(64), 
            nn.BatchNorm2d(64), 
            nn.BatchNorm2d(64), 
            nn.BatchNorm2d(128)])

        # self.rnn_1 = nn.RNN(128, 64, bidirectional=True)
        # self.rnn_2 = nn.RNN(128, 64, bidirectional=True)
        # self.rnn_3 = nn.RNN(128, 32, bidirectional=True)
        # self.rnn_4 = nn.RNN(64, 32, bidirectional=True)

        # output_T = 19
        # output_T = 31
        output_T = 2

        self.dropout_p = 0.2

        # m1: [256 x 1984], m2: [1216 x 10]
        # m1: [256 x 5824], m2: [1984 x 992]

        self.fc = nn.Linear(64 * output_T, 32 * output_T)
        self.fc_2 = nn.Linear(32 * output_T, 16 * output_T)
        self.fc_3 = nn.Linear(16 * output_T, len(SCENE_DICT))
        # self.fc = nn.Linear(128 * 19, len(SCENE_DICT))
        # (B, M[80], H[1], T[101]) => (B, M[80], H[256], T[21])
        # self.fc_device = nn.Linear(128 * 19, len(DEVICE_DICT))
        # self.fc_city = nn.Linear(128 * 19, len(CITY_DICT))

        self.fc_device = nn.Linear(64 * output_T, len(DEVICE_DICT))
        self.fc_city = nn.Linear(64 * output_T, len(CITY_DICT))
        
    def forward(self, tensor):
        tensor = tensor.unsqueeze(1)

        # print(tensor.shape)

        for conv_layer, batch_norm in zip(self.conv_layers, self.batch_norm_layers):
            tensor = F.relu(batch_norm(conv_layer(tensor)))
            # print(tensor.shape)

        # tensor = tensor.squeeze(2)
        # tensor = tensor.permute(2, 0, 1) # (B, H, T) => (T, B, H)
        # tensor, _ = self.rnn_1(tensor)
        # tensor = F.dropout(tensor, self.dropout_p)

        # tensor, _ = self.rnn_2(tensor)
        # tensor = F.dropout(tensor, self.dropout_p)

        # tensor, _ = self.rnn_3(tensor)
        # tensor = F.dropout(tensor, self.dropout_p)

        # tensor, _ = self.rnn_4(tensor)
        # tensor = F.dropout(tensor, self.dropout_p)
        # tensor = tensor.permute(1, 2, 0) # (T, B, H) => (B, H, T)
        tensor = tensor.reshape(tensor.shape[0], -1)
        tensor_scene = self.fc(tensor)
        tensor_scene = F.dropout(tensor_scene, self.dropout_p)
        tensor_scene = self.fc_2(tensor_scene)
        tensor_scene = F.dropout(tensor_scene, self.dropout_p)
        tensor_scene = self.fc_3(tensor_scene)
        tensor_scene = F.log_softmax(tensor_scene, dim=-1)

        tensor_device = self.fc_device(tensor)
        tensor_device = F.log_softmax(tensor_device, dim=-1)

        tensor_city = self.fc_city(tensor)
        tensor_city = F.log_softmax(tensor_city, dim=-1)

        return tensor_scene, tensor_city, tensor_device


if __name__ == '__main__':

    print('{:^43s}'.format('==== Display hparams ===='))
    display_dict(hparams)
    
    dataset_name = 'TAU-urban-acoustic-scenes-2020-mobile-development'

    dataset_base_path = os.path.join('datasets', dataset_name)

    run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_june_bn_deeper_delta'

    log_path = os.path.join('logs', run_name)
    os.makedirs(log_path, exist_ok=True)

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

    tensor_writer = SummaryWriter(log_path)

    net = CRNN_Net_2().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_steps = 0
    eval_steps = 0

    loss_scene_train_avg = Averager()
    acc_scene_train_avg = Averager()
    loss_device_train_avg = Averager()
    acc_device_train_avg = Averager()
    loss_city_train_avg = Averager()
    acc_city_train_avg = Averager()

    loss_scene_eval_avg = Averager()
    acc_scene_eval_avg = Averager()
    loss_device_eval_avg = Averager()
    acc_device_eval_avg = Averager()
    loss_city_eval_avg = Averager()
    acc_city_eval_avg = Averager()

    last_loss_scene_train_avg = 0
    last_acc_scene_train_avg = 0
    last_loss_device_train_avg = 0
    last_acc_device_train_avg = 0
    last_loss_city_train_avg = 0
    last_acc_city_train_avg = 0

    for epoch in range(hparams['max_epoch']):

        print('[ Training Epoch #{:03d} ]'.format(epoch))
        train_dataset_loader.start_loading()
        for i, batch in tqdm(enumerate(train_dataset_loader.generator()), total=train_dataset_loader.max_batch_size):
            
            mel_spectrogram_batch, file_path_list, city_label, device_label, scene_label = batch
            optimizer.zero_grad()
            outputs_scene, outputs_city, outputs_device = net(mel_spectrogram_batch.to(device))
            loss_scene = criterion(outputs_scene, torch.tensor(scene_label, dtype=torch.int64).to(device))
            loss_city = criterion(outputs_city, torch.tensor(city_label, dtype=torch.int64).to(device))
            loss_device = criterion(outputs_device, torch.tensor(device_label, dtype=torch.int64).to(device))
            loss = loss_scene # + loss_city / 10 + loss_device / 10
            loss.backward()
            optimizer.step()

            pred_scene = torch.argmax(outputs_scene, 1)
            acc_scene = torch.sum(pred_scene == torch.tensor(scene_label).to(device)).cpu().numpy() / len(scene_label)

            pred_city = torch.argmax(outputs_city, 1)
            acc_city = torch.sum(pred_city == torch.tensor(city_label).to(device)).cpu().numpy() / len(city_label)

            pred_device = torch.argmax(outputs_device, 1)
            acc_device = torch.sum(pred_device == torch.tensor(device_label).to(device)).cpu().numpy() / len(device_label)

            loss_scene_train_avg += loss_scene.item()
            acc_scene_train_avg += acc_scene
            loss_device_train_avg += loss_device.item()
            acc_device_train_avg += acc_device
            loss_city_train_avg += loss_city.item()
            acc_city_train_avg += acc_city

            # print(loss.item())
            if train_steps % hparams['logging_steps'] == 0:
                tensor_writer.add_scalar('train/loss_scene', loss_scene_train_avg.value, train_steps)
                tensor_writer.add_scalar('train/acc_scene', acc_scene_train_avg.value, train_steps)
                tensor_writer.add_scalar('train/loss_device', loss_device_train_avg.value, train_steps)
                tensor_writer.add_scalar('train/acc_device', acc_device_train_avg.value, train_steps)
                tensor_writer.add_scalar('train/loss_city', loss_city_train_avg.value, train_steps)
                tensor_writer.add_scalar('train/acc_city', acc_city_train_avg.value, train_steps)

                last_loss_scene_train_avg = loss_scene_train_avg.value
                last_acc_scene_train_avg = acc_scene_train_avg.value
                last_loss_device_train_avg = loss_device_train_avg.value
                last_acc_device_train_avg = acc_device_train_avg.value
                last_loss_city_train_avg = loss_city_train_avg.value
                last_acc_city_train_avg = acc_city_train_avg.value

                loss_scene_train_avg.reset()
                acc_scene_train_avg.reset()
                loss_device_train_avg.reset()
                acc_device_train_avg.reset()
                loss_city_train_avg.reset()
                acc_city_train_avg.reset()

            train_steps += 1

            # break

            # print(batch[0].shape)

        print('[ Evaluation Epoch #{:03d} ]'.format(epoch))
        eval_dataset_loader.start_loading()
        for i, batch in tqdm(enumerate(eval_dataset_loader.generator()), total=eval_dataset_loader.max_batch_size):

            mel_spectrogram_batch, file_path_list, city_label, device_label, scene_label = batch
            outputs_scene, outputs_city, outputs_device = net(mel_spectrogram_batch.to(device))
            loss_scene = criterion(outputs_scene, torch.tensor(scene_label, dtype=torch.int64).to(device))
            loss_city = criterion(outputs_city, torch.tensor(city_label, dtype=torch.int64).to(device))
            loss_device = criterion(outputs_device, torch.tensor(device_label, dtype=torch.int64).to(device))

            pred_scene = torch.argmax(outputs_scene, 1)
            acc_scene = torch.sum(pred_scene == torch.tensor(scene_label).to(device)).cpu().numpy() / len(scene_label)

            pred_city = torch.argmax(outputs_city, 1)
            acc_city = torch.sum(pred_city == torch.tensor(city_label).to(device)).cpu().numpy() / len(city_label)

            pred_device = torch.argmax(outputs_device, 1)
            acc_device = torch.sum(pred_device == torch.tensor(device_label).to(device)).cpu().numpy() / len(device_label)

            loss_scene_eval_avg += loss_scene.item()
            acc_scene_eval_avg += acc_scene
            loss_device_eval_avg += loss_device.item()
            acc_device_eval_avg += acc_device
            loss_city_eval_avg += loss_city.item()
            acc_city_eval_avg += acc_city

            if eval_steps % hparams['logging_steps'] == 0:
                tensor_writer.add_scalar('eval/loss_scene', loss_scene_eval_avg.value, eval_steps)
                tensor_writer.add_scalar('eval/acc_scene', acc_scene_eval_avg.value, eval_steps)
                tensor_writer.add_scalar('eval/loss_device', loss_device_eval_avg.value, eval_steps)
                tensor_writer.add_scalar('eval/acc_device', acc_device_eval_avg.value, eval_steps)
                tensor_writer.add_scalar('eval/loss_city', loss_city_eval_avg.value, eval_steps)
                tensor_writer.add_scalar('eval/acc_city', acc_city_eval_avg.value, eval_steps)

                tensor_writer.add_scalar('delta/loss_scene', loss_scene_eval_avg.value - last_loss_scene_train_avg, eval_steps)
                tensor_writer.add_scalar('delta/acc_scene', acc_scene_eval_avg.value - last_acc_scene_train_avg, eval_steps)
                tensor_writer.add_scalar('delta/loss_device', loss_device_eval_avg.value - last_loss_device_train_avg, eval_steps)
                tensor_writer.add_scalar('delta/acc_device', acc_device_eval_avg.value - last_acc_device_train_avg, eval_steps)
                tensor_writer.add_scalar('delta/loss_city', loss_city_eval_avg.value - last_loss_city_train_avg, eval_steps)
                tensor_writer.add_scalar('delta/acc_city', acc_city_eval_avg.value - last_acc_city_train_avg, eval_steps)

                loss_scene_eval_avg.reset()
                acc_scene_eval_avg.reset()
                loss_device_eval_avg.reset()
                acc_device_eval_avg.reset()
                loss_city_eval_avg.reset()
                acc_city_eval_avg.reset()

            eval_steps += 1

            # break
