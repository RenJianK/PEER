#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File : process_highDPredictData.py
# @Project : KDGAN-CSR
# @Author : RenJianK
# @Time : 2022/11/7 16:07

import json
import math
import os
import numpy as np

from config import Config
from data_process.prepare_highD import PrepareData
from data_process.traffic_flow import TrafficFlow


class PredictData(object):
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
    """

    def __init__(self):
        self.config = Config()  # basic configuration
        self.original_data = PrepareData(load=True)
        self.traffic_flow = TrafficFlow()  # traffic flow
        self.grid_length = self.config.min_gap + self.config.min_length  # grid length
        self.max_length = self.config.max_road_length  # maximal length
        self.max_lane = self.config.max_lane_num  # maximal lane

        self.predict_train_data_file = os.path.join(
            self.config.path.data_path, self.config.predict_train_data_path
        )

    def save(self, data):
        with open(self.predict_train_data_file, 'w') as f:
            json.dump(data, f)

    def process_data(self):
        # generate the train data of the group trajectory predict model
        # the construction of data is [
        # [
        #    "road_name": {
        #       "grid": grid,
        #       "vid": vid
        #    }
        # ], ...]

        # 1. First, load the road state data
        vehicle_file = os.path.join(
            self.config.path.data_path,
            self.config.vehicle_filename
        )
        with open(vehicle_file, 'r') as f:
            all_information = json.load(f)

        # 2. Second, initial the data storage variable
        data = []

        # 3. Third, traverse data
        i = 0
        for t, val in enumerate(all_information):
            # compute the global grid map
            i += 1
            if i % 100 == 0:
                print("data process step {} | {}".format(i, len(all_information)))

            instantaneous = {}
            roads = val['vehicles']

            for road in roads:
                vehicles = roads[road]['vehicles']
                road_data = self.grid_map(t, road, vehicles)

                instantaneous[road] = road_data

            data.append(instantaneous)

        self.save(data)

    def grid_map(self, t, road, vehicles):
        # Generate grid map, which contains [x, y, lane, s, c, a, l, f]
        l, n = self.max_length, self.max_lane
        road_shape = self.original_data.get_road_shape(road)  # get the road shape of lanes

        v_number, h_number = int(self.max_lane), math.ceil(self.max_length / self.grid_length)
        l_number = math.ceil(l / self.grid_length)
        m = np.zeros((h_number, v_number, 9))  # storage grid information
        v = ['' for _ in range(h_number * v_number)]  # storage the vehicle id

        if v_number > n:
            m[:, n:][:] = -1

        if self.max_length > l:
            m[l_number:, :][:] = -1

        # initialize grid map
        for i in range(h_number):
            for j in range(v_number):
                x, lane = i * self.grid_length, j
                y = (road_shape[j] + road_shape[j + 1]) / 2
                f = self.traffic_flow.get_traffic_flow (
                    t, road)
                m[i, j] = [x, y, lane, 0, 0, 0, self.config.max_road_length - x, self.grid_length, f]

        # fill the vehicle state into the grid occupied by vehicle,
        # the multi occupied grid will be filled for long vehicle
        for vehicle in vehicles:
            info = vehicles[vehicle]['info']
            y = info['lateral_position']
            speed = info['speed']
            lane = int(info['lane'])
            length = info['length']
            position_in_lane = info['position_in_lane']
            accelerate = info['acceleration']

            h_p = math.floor(position_in_lane / self.grid_length)
            h_p = h_p if h_p < h_number else h_number - 1
            occupied_grid_number = 1 + length // self.grid_length
            if occupied_grid_number % 2 != 0:
                start_grid = h_p - occupied_grid_number // 2
                end_grid = h_p + occupied_grid_number // 2 + 1
            else:
                start_grid = h_p - occupied_grid_number // 2 if position_in_lane % self.grid_length > (
                        self.grid_length // 2) else h_p + 1 - occupied_grid_number // 2
                end_grid = start_grid + occupied_grid_number

            start_grid, end_grid = int(start_grid), int(end_grid)
            end_grid = end_grid if end_grid < h_number else h_number
            start_grid = start_grid if start_grid > 0 else 0

            for index in range(start_grid, end_grid):
                m[index, lane][:8] = [
                    position_in_lane, y, lane, speed, 1, accelerate,
                    self.config.max_road_length - position_in_lane,
                    length
                ]

                v[index * v_number + lane] = vehicle

        return {
            "grid": m.tolist(),
            "vid": v
        }


if __name__ == '__main__':
    pre = PredictData()
    pre.process_data()
    print()
