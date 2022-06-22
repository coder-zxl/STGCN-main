import torch.nn.functional as F
import torch

from torch.utils.data import DataLoader
from typing import Tuple

# from datasets.utils.ST_continual_dataset import STContinualDataset, store_sequential_loaders
from script.ST_continual_dataset import *

class Sequential_METR_LA(STContinualDataset):
    NAME = 'seq_metr_la'
    SETTING = 'instance-il'
    N_hop_per_task = 1
    N_TASKS = 5

    def get_data_loaders(self):
        train_list, val_list, test_list, scaler = generate_seq_dataset(file='./data/metr-la/vel.csv')
        train_dataset_list = []
        val_dataset_list = []
        test_dataset_list = []
        for i in range(len(train_list)):
            x_train, y_train = train_list[i]
            flag_train_dataset = STDataset(x_train, y_train)
            train_dataset_list.append(flag_train_dataset)
            # x_data = flag_train_dataset.input
            # y_data = flag_train_dataset.truth

            x_val, y_val = val_list[i]
            flag_val_dataset = STDataset(x_val, y_val)
            val_dataset_list.append(flag_val_dataset)



            x_test, y_test = test_list[i]
            flag_test_dataset = STDataset(x_test, y_test)
            test_dataset_list.append(flag_test_dataset)

            # print('Task: {}, x_train shape: {}, y_train shape: {}'.format(i+1, x_train.shape, y_train.shape))
            # print('x_val shape: {}, y_val shape: {}'.format(x_val.shape, y_val.shape))
            # print('x_test shape: {}, y_test shape: {}'.format(x_test.shape, y_test.shape))
            # print('num_samples: {}, num_samples+num_test: {}'.format(num_samples, num_samples+num_test))
            # print()
        
        train_loader, val_loader, test_loader = store_sequential_loaders(train_dataset_list, val_dataset_list, test_dataset_list, self)
        return train_loader, val_loader, test_loader, scaler


args = None
metr_la_dataset = Sequential_METR_LA(args)

# for t in range(metr_la_dataset.N_TASKS):
#     train_loader, val_loader, test_loader, scaler = metr_la_dataset.get_data_loaders()
#     print('Task {} end!!!'.format(t))