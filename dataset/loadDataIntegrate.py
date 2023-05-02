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
from data_process.process_data_integrate import ProcessData

random.seed(333)


def integrate_data(inputs, real, train, distance_index):
    res_i = []
    res_r = []

    for i in range(inputs):
        index = [i for i in range(len(inputs[i]))]
        random.shuffle(index)
        index = index[len(index) * 1 // 5:len(index) * 4 // 5]

        if train:
            res_i.append(inputs[index])



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

    def __init__(self, train=True, test_distance=0):
        self.train = train
        self.config = config.Config()
        index = [0.5 for _ in range(10)]
        self.normal = Normalize(index, index)

        self.v2x_data_file = os.path.join(
            self.config.path.data_path, self.config.v2x_data
        )

        if not os.path.isfile(self.v2x_data_file):
            process_data = ProcessData(500, 3)
            process_data.process_data()

        self.data = []

        with open(self.v2x_data_file, 'r') as f:
            self.v2x_data = json.load(f)

        self.input, self.real = self.process()
        self.index = [i for i in range(len(self.input))]
        random.shuffle(self.index)
        self.index = self.index[len(self.index) * 1 // 5:len(self.index) * 4 // 5]

        self.diff = int(len(self.index) * 0.8)
        self.length = self.diff if train else len(self.index) - self.diff

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        item = item if self.train else item + self.diff
        return self.input[self.index[item]], self.real[self.index[item]]

    def process(self):
        # generate data with history info
        i_data = []
        r_data = []
        for distance in self.v2x_data:
            input_data = []
            real_data = []

            for key in range(self.config.history_window, len(self.v2x_data[distance])):
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

            i_data.append(input_data)
            r_data.append(real_data)

        return i_data, r_data


if __name__ == '__main__':
    load = LoadData()
    print(load[0, 0])
