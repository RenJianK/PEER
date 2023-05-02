#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File : loadData.py
# @Project : KDGAN-CSR
# @Author : RenJianK
# @Time : 2022/5/16 15:48
import os
import json
import random

import torch
from torchvision.transforms import Normalize

import config
from data_process.process_data import ProcessData

random.seed(333)


class LoadData(object):
    # ---- load data with history info ----
    """
    [
        road: {
            'global': g_m,
            'v2x': v_m,
            'useful': label
        }
    ]
    """

    def __init__(self, train=True):
        self.train = train
        self.config = config.Config()
        index = [0.5 for _ in range(10)]
        self.normal = Normalize(index, index)

        self.v2x_data_file = os.path.join(
            self.config.path.data_path, self.config.v2x_data
        )

        # if train:
        #     self.v2x_data_file = os.path.join(
        #         self.config.path.data_path, self.config.v2x_train_data
        #     )
        # else:
        #     self.v2x_data_file = os.path.join(
        #         self.config.path.data_path, self.config.v2x_test_data
        #     )

        if not os.path.isfile(self.v2x_data_file):
            process_data = ProcessData(50, 500, 3)
            process_data.process_data()

        self.data = []

        with open(self.v2x_data_file, 'r') as f:
            self.v2x_data = json.load(f)

        self.input, self.real = self.process()
        self.index = [i for i in range(len(self.input))]
        random.shuffle(self.index)
        self.index = self.index[len(self.index) * 1 // 5:len(self.index) * 4 // 5]
        # [len(self.index) * 2 // 5:len(self.index) * 4 // 5]

        self.diff = int(len(self.index) * 0.8)
        self.length = self.diff if train else len(self.index) - self.diff
        # print(len(self.input))

        # self.input = torch.tensor(self.input)
        # s_0, s_1, s_2, s_3 = self.input.shape[0], self.input.shape[1], self.input.shape[2], self.input.shape[3]
        # self.input = torch.reshape(self.input, (-1, self.input.shape[2], self.input.shape[3], self.input.shape[4]))
        # self.input = torch.reshape(self.input, (self.input.shape[0], -1, self.input.shape[3]))
        # print(self.input.shape)

        # self.input = self.normal(self.input)
        # print(self.input.shape)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        item = item if self.train else item + self.diff
        return self.input[self.index[item]], self.real[self.index[item]]

    def process(self):
        # generate data with history info
        input_data = []
        real_data = []

        for key in range(self.config.history_window, len(self.v2x_data)):
            roads = self.v2x_data[key]
            # it is useful if the state of 'useful' in the road is set to 1
            for road in roads:
                if roads[road]['useful'] == 0:
                    continue

                real_data.append(roads[road]['global'])
                history_data = []
                for i in range(key - self.config.history_window + 1, key):
                    history_data.append(self.v2x_data[i][road]['v2x'])

                history_data.append(roads[road]['v2x'])
                input_data.append(history_data)

        # Normalise the data
        # input_data = torch.tensor(input_data, dtype=torch.float32)  # [n, 5, 40, 3, 10]
        # input_data = input_data.reshape(
        #     (input_data.shape[0] * input_data.shape[1], input_data.shape[2], input_data.shape[3], -1))
        # input_data = input_data.permute((0, 3, 1, 2))
        # input_data = self.normal(input_data)
        # input_data = input_data.reshape(
        #     (input_data.shape[0] // 5, 5, input_data.shape[1], input_data.shape[2], input_data.shape[3]))
        # input_data = input_data.permute((0, 1, 3, 4, 2))
        # print(input_data.shape)
        return input_data, real_data


if __name__ == '__main__':
    load = LoadData()
    print(load[0, 0])
