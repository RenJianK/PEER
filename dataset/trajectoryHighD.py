#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File : trajectoryHighD.py
# @Project : KDGAN-CSR
# @Author : RenJianK
# @Time : 2022/11/10 20:40

import json
import os
import random

import config
from data_process.process_highDPredictData import PredictData
from tool import getFutureVehicleHighD

random.seed(333)


class TrajectoryState(object):
    """Summary of class here.

    Longer class information....

    Args:
        beam_size (int): Number of beams to use (see base ``parallel_paths``).
        batch_size (int): See base.
        pad (int): See base.
        bos (int): See base.
        eos (int): See base.

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.

    the construction of data is [
        {
            "road_name": {
               "grid": grid,
               "vid": vid
            },
            ...
        },
        ...
    ]
    """

    def __init__(self, train=True):
        self.train = train
        self.config = config.Config()

        self.predict_train_data_file = os.path.join(
            self.config.path.data_path, self.config.predict_train_data_path
        )

        if not os.path.isfile(self.predict_train_data_file):
            process_data = PredictData()
            process_data.process_data()

        self.data = []

        with open(self.predict_train_data_file, 'r') as f:
            self.predict_data = json.load(f)

        self.input = self.process()
        self.index = [i for i in range(len(self.input))]
        random.shuffle(self.index)
        self.index = self.index

        self.diff = int(len(self.index) * 0.8)
        self.length = self.diff if train else len(self.index) - self.diff

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        item = item if self.train else item + self.diff
        return self.input[self.index[item]]['grid'], \
               self.input[self.index[item]]['x'], \
               self.input[self.index[item]]['y'], self.input[self.index[item]]['n']

    def process(self):
        # generate data with history info
        input_data = []
        for key in range(self.config.history_window, len(self.predict_data)):
            instantaneous = self.predict_data[key]
            # it is useful if the state of 'useful' in the road is set to 1
            for road in instantaneous:
                history_data = []
                for i in range(key - self.config.history_window + 1, key):
                    history_data.append(self.predict_data[i][road]['grid'])

                history_data.append(instantaneous[road]['grid'])

                x, y, n = getFutureVehicleHighD(
                    history_data[-1],
                    instantaneous[road]['vid'],
                    self.predict_data[key - 1][road]['vid']
                )

                input_data.append({
                    'grid': history_data,
                    'vid-now': instantaneous[road]['vid'],
                    'vid-before': self.predict_data[key - 1][road]['vid'],
                    'rid': road,
                    'x': x,
                    'y': y,
                    'n': n,
                    't': key
                })

        return input_data


if __name__ == '__main__':
    ts = TrajectoryState()
    print(ts[0][1])
    print(ts[0][2])
    print(ts[0][-1])
