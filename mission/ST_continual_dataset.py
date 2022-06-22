# Construction of spatio-temporal continual dataset

# from tkinter.messagebox import NO
from tkinter.messagebox import NO
import numpy as np
import pandas as pd
import torch.utils.data as Data

from abc import abstractmethod
from argparse import Namespace
from torch import nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
# from data.get_logger import get_logger


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def transform(self, data):
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class STContinualDataset:
    NAME = None
    SETTING = None
    N_hop_per_task = None
    N_TASKS = None

    def __init__(self, args: Namespace) -> None:
        self.train_loader = None
        self.test_loaders = []
        # self.memory_loaders = []
        self.val_loaders = []
        self.train_loaders = []
        self.N_hop_per_task = 1
        self.i = 0
        self.args = args
    
    @abstractmethod
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        pass

    @abstractmethod
    def not_aug_dataloader(self, batch_size: int) -> DataLoader:
        pass

    @staticmethod
    @abstractmethod
    def get_backbone() -> nn.Module:
        pass

    @staticmethod
    @abstractmethod
    def get_loss() -> nn.functional:
        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform():
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform():
        pass


def generate_graph_seq2seq_io_data(df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None):
    # Return
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]

    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, 'D')
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), : df.index.dayofweek] = 1
        data_list.append(day_in_week)
    
    data = np.concatenate(data_list, axis=-1)

    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x_t = data[t+x_offsets, ...]
        y_t = data[t+y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_seq_dataset(file='data/metr-la.h5'):
    df = pd.read_hdf(file)
    x_offsets = np.sort(
        np.concatenate((np.arange(-11, 1, 1),))
    )
    y_offsets = np.sort(np.arange(1, 13, 1))

    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False
    )

    n_tasks = 5

    num_all = x.shape[0]
    num_samples = int(num_all/3)
    num_test = int((num_all-num_samples)/n_tasks)

    scaler = StandardScaler(mean=x[..., 0].mean(), std=x[..., 0].std())

    x[..., 0] = scaler.transform(x[..., 0])
    # y[..., 0] = scaler.transform(y[..., 0])

    train_list = []
    val_list = []
    test_list = []

    for i in range(n_tasks):
        if i == 0:
            x_flag, y_flag = x[:num_samples], y[:num_samples]
            train_len = int(len(x_flag)*0.8)
            x_train, y_train = x_flag[:train_len], y_flag[:train_len]
            x_val, y_val = x_flag[train_len:], y_flag[train_len:]
            x_test, y_test = x[num_samples:num_samples+num_test], y[num_samples:num_samples+num_test]
            # print('Task: {}, x_train shape: {}, y_train shape: {}'.format(i+1, x_train.shape, y_train.shape))
            # print('x_val shape: {}, y_val shape: {}'.format(x_val.shape, y_val.shape))
            # print('x_test shape: {}, y_test shape: {}'.format(x_test.shape, y_test.shape))
            # print('num_samples: {}, num_samples+num_test: {}'.format(num_samples, num_samples+num_test))
            # print()

            # train_loader = DataLoader(x_train, y_train, batch_size, shuffle=True)
            # val_loader = DataLoader(x_val, y_val, batch_size, shuffle=False)
            # test_loader = DataLoader(x_test, y_test, batch_size, shuffle=False)

            train_list.append((x_train, y_train))
            val_list.append((x_val, y_val))
            test_list.append((x_test, y_test))
        else:
            x_flag, y_flag = x[num_samples+(i-1)*num_test:num_samples+i*num_test], y[num_samples+(i-1)*num_test:num_samples+i*num_test]
            train_len = int(len(x_flag)*0.8)
            x_train, y_train = x_flag[:train_len], y_flag[:train_len]
            x_val, y_val = x_flag[train_len:], y_flag[train_len:]
            x_test, y_test = x[num_samples+i*num_test:num_samples+(i+1)*num_test], y[num_samples+i*num_test:num_samples+(i+1)*num_test]

            # print('Task: {}, x_train shape: {}, y_train shape: {}'.format(i+1, x_train.shape, y_train.shape))
            # print('x_val shape: {}, y_val shape: {}'.format(x_val.shape, y_val.shape))
            # print('x_test shape: {}, y_test shape: {}'.format(x_test.shape, y_test.shape))
            # print('num_samples + i*num_test: {}, num_samples+(i+1)num_test: {}'.format(num_samples+i*num_test, num_samples+(i+1)*num_test))
            # print()

            # train_loader = DataLoader(x_train, y_train, batch_size, shuffle=True)
            # val_loader = DataLoader(x_val, y_val, batch_size, shuffle=False)
            # test_loader = DataLoader(x_test, y_test, batch_size, shuffle=False)

            train_list.append((x_train, y_train))
            val_list.append((x_val, y_val))
            test_list.append((x_test, y_test))
        
    return train_list, val_list, test_list, scaler


class STDataset(Dataset):
    def __init__(self, x_data, y_data):
        # self.mode = mode
        self.input = x_data
        self.truth = y_data
        # if self.mode == 'train':
        #     self.input = x_data['train']
        #     self.truth = y_data['train']
        # elif self.mode == 'val':
        #     self.input = x_data['val']
        #     self.truth = y_data['val']
        # elif self.mode == 'test':
        #     self.input = x_data['test']
        #     self.truth = y_data['test']
        # else:
        #     print('Mode Error!!!!')
    
    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, index):
        self.sample_x = self.input[index]
        self.sample_y = self.truth[index]
        return self.sample_x, self.sample_y


def store_sequential_loaders(train_dataset_list, val_dataset_list, test_dataset_list, setting: STContinualDataset):
    print('Setting i: {}'.format(setting.i))
    train_dataset = train_dataset_list[setting.i]
    val_dataset = val_dataset_list[setting.i]
    test_dataset = test_dataset_list[setting.i]

    x_train, y_train = train_dataset.input, train_dataset.truth
    x_val, y_val = val_dataset.input, val_dataset.truth
    x_test, y_test = test_dataset.input, test_dataset.truth

    print('Task: {}, x_train shape: {}, y_train shape: {}'.format(setting.i, x_train.shape, y_train.shape))
    print('x_val shape: {}, y_val shape: {}'.format(x_val.shape, y_val.shape))
    print('x_test shape: {}, y_test shape: {}'.format(x_test.shape, y_test.shape))
    # print('num_samples: {}, num_samples+num_test: {}'.format(num_samples, num_samples+num_test))
    # print()

    # train_loader = DataLoader(train_dataset, batch_size=setting.args.batch_size, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=setting.args.batch_size, shuffle=False, num_workers=4)
    # test_loader = DataLoader(test_dataset, batch_size=setting.args.batch_size, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    setting.train_loaders.append(train_loader)
    setting.val_loaders.append(val_loader)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += setting.N_hop_per_task
    return train_loader, val_loader, test_loader

#    